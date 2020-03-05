import json
from collections import defaultdict

import numpy as np
import fire


def read_label(input_file):
    with open(input_file, 'r') as fp:
        objs = {}
        for line in fp.readlines():
            obj = json.loads(line)
            uuid = obj['uuid']
            label = obj['data']
            if label not in set(["Not tobacco related", "Tobacco related"]):
                continue
            if label == "Not tobacco related":
                label = {'about_tobacco': 'no'}
            elif label == "Tobacco related":
                label = {'about_tobacco': 'yes'}
            else:
                raise ValueError
            objs[uuid] = label
    return objs


class Main:
    def split(self,
              data_file='data/tobacco/mongo_feedback_new.json',
              label_file='data/tobacco/feedback_new.json',
              out_file='data/tobacco/clean_feedback_new.json',
              ratio='8:1:1'):
        label = read_label(label_file)
        ratio = [float(x) for x in ratio.split(':')]
        train = ratio[0] / sum(ratio)
        dev = ratio[1] / sum(ratio)
        test = ratio[2] / sum(ratio)
        data = defaultdict(list)
        with open(data_file, 'r') as fp, open(out_file, 'w') as out:
            for line in fp.readlines():
                obj = json.loads(line)
                if obj['uuid'] not in label:
                    continue
                clean_obj = {}
                clean_obj['fold'] = None
                clean_obj['lang'] = obj['language']
                clean_obj['title'] = obj['title']
                clean_obj['body'] = obj['cleaned_content']
                clean_obj['label'] = label[obj['uuid']]
                data[clean_obj['lang']].append(clean_obj)

            for lang in data:
                objs = data[lang]
                np.random.shuffle(objs)
                train_ = int(train * len(objs))
                dev_ = int(dev * len(objs))
                train_set = objs[:train_]
                dev_set = objs[train_:train_ + dev_]
                test_set = objs[train_ + dev_:]
                print(lang, len(train_set), len(dev_set), len(test_set),
                      len(objs))
                for obj in train_set:
                    obj['fold'] = 'train'
                    print(json.dumps(obj), file=out)
                for obj in dev_set:
                    obj['fold'] = 'dev'
                    print(json.dumps(obj), file=out)
                for obj in test_set:
                    obj['fold'] = 'test'
                    print(json.dumps(obj), file=out)

    def test_all(self,
                 data_file='data/tobacco/mongo_feedback.json',
                 label_file='data/tobacco/feedback.json',
                 out_file='data/tobacco/clean_feedback.json'):
        label = read_label(label_file)
        with open(data_file, 'r') as fp, open(out_file, 'w') as out:
            for line in fp.readlines():
                obj = json.loads(line)
                if obj['uuid'] not in label:
                    continue
                clean_obj = {}
                clean_obj['fold'] = 'test'
                clean_obj['lang'] = obj['language']
                clean_obj['title'] = obj['title']
                clean_obj['body'] = obj['cleaned_content']
                clean_obj['label'] = label[obj['uuid']]
                print(json.dumps(clean_obj), file=out)

    def extend_split(self,
                     prefix='data/tobacco/curated-',
                     split_file='data/tobacco/clean_feedback_new.json',
                     original_file='data/tobacco/original/mongo_feedback_new.json'):
        data = dict()
        with open(original_file, 'r') as fp:
            for line in fp.readlines():
                obj = json.loads(line)
                try:
                    extend_obj = {}
                    if obj['language'] == 'en':
                        extend_obj['body_original_lang'] = obj['cleaned_content']
                        extend_obj['title_original_lang'] = obj['title']
                    else:
                        extend_obj['body_original_lang'] = obj['cleaned_content_translated']
                        extend_obj['title_original_lang'] = obj['title_translated']
                    extend_obj['body_translated'] = obj['cleaned_content']
                    extend_obj['title_translated'] = obj['title']
                    extend_obj['url'] = obj['url']
                    extend_obj['uuid'] = obj['uuid']
                    data[obj['title']] = extend_obj
                except:
                    print('skip', obj['uuid'])

        lang_fps = dict()
        with open(split_file, 'r') as fp:
            for line in fp.readlines():
                obj = json.loads(line)
                lang = obj['lang']
                if lang in lang_fps:
                    out = lang_fps[lang]
                else:
                    out = open(prefix + lang, 'w')
                    lang_fps[lang] = out
                extend_obj = data[obj['title']]
                for field in extend_obj:
                    obj[field] = extend_obj[field]
                obj['label']['relevance'] = obj['label']['about_tobacco']
                del obj['lang']
                print(json.dumps(obj), file=out)


if __name__ == "__main__":
    fire.Fire(Main)
