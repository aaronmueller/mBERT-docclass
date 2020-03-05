# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning labeling."""

from collections import namedtuple
from dataclasses import dataclass
from typing import List

import torch
from pytorch_transformers import BertTokenizer, XLMTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import processor
from module import DecoderConfigFactory, EncoderConfigFactory, SeqLabelConfig
from module.evaluator import NEREvaluator, POSEvaluator
from trainer import BaseTrainer
from util import Mode, maybe_mkdir


@dataclass
class InputFeatures:
    """A single set of features of data."""
    input_ids: List[int]
    segment_ids: List[int]
    position_ids: List[int]
    input_mask: List[int]
    label_ids: List[int]


@dataclass
class SeqSpan:
    start: int
    end: int


def examples_to_dataloader(trainer,
                           examples,
                           language,
                           max_seq_length,
                           seq_stride,
                           batch_size,
                           evaluate,
                           get_raw_data=False):
    """Loads a data file into a list of `InputBatch`s."""
    logger = trainer.logger
    label_map = {
        label: idx
        for (idx, label) in enumerate(trainer.processor.get_labels())
    }
    label_map[-1] = -1  # pad label

    if isinstance(trainer.tokenizer, XLMTokenizer):
        try:
            seg_id = trainer.tokenizer.lang2id[language]
        except:
            seg_id = 0
    elif isinstance(trainer.tokenizer, BertTokenizer):
        seg_id = 0
    else:
        raise ValueError(trainer.tokenizer)

    # Step 1: tokenize examples
    data = []
    for example in examples:
        all_tokens = []
        all_labels = []
        assert len(example.text) == len(example.label)
        for (i, (token, label)) in enumerate(zip(example.text, example.label)):
            if isinstance(trainer.tokenizer, XLMTokenizer):
                sub_tokens = trainer.tokenizer.tokenize(token, lang=language)
            elif isinstance(trainer.tokenizer, BertTokenizer):
                sub_tokens = trainer.tokenizer.tokenize(token)
            else:
                raise ValueError(trainer.tokenizer)
            if len(sub_tokens) >= 1:
                all_tokens.append(sub_tokens[0])
                all_labels.append(label)
            if len(sub_tokens) > 1:
                for sub_token in sub_tokens[1:]:
                    all_tokens.append(sub_token)
                    # pad label, ignore when computing loss
                    all_labels.append(-1)
        data.append((example, all_tokens, all_labels))
    if get_raw_data:
        return data

    emperical_max_len = max([len(d[1]) for d in data]) + 2
    max_seq_length = min(max_seq_length, emperical_max_len)
    logger.info(f'Emperical seq len {max_seq_length}')

    # Step 2: create tensors
    features: List[InputFeatures] = []
    for (ex_index, d) in enumerate(data):
        example, all_tokens, all_labels = d
        # We can have sequences that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `seq_stride`.
        seq_spans = []
        # Account for [CLS] and [SEP] with "- 2"
        max_tokens_for_seq = max_seq_length - 2
        start_offset = 0
        while start_offset < len(all_tokens):
            length = len(all_tokens) - start_offset
            if length > max_tokens_for_seq:
                length = max_tokens_for_seq
            seq_spans.append(
                SeqSpan(start=start_offset, end=start_offset + length))
            if start_offset + length == len(all_tokens):
                break
            start_offset += min(length, seq_stride)

        for span_idx, seq_span in enumerate(seq_spans):
            tokens: List[str] = [trainer.tokenizer.cls_token]
            label_ids: List[int] = [-1]

            tok_cnt = 0
            for token, label in zip(all_tokens[seq_span.start:seq_span.end],
                                    all_labels[seq_span.start:seq_span.end]):
                # already predicted in previous span
                if tok_cnt < seq_stride and span_idx > 0:
                    label_ids.append(-1)
                else:
                    label_ids.append(label_map[label])
                tokens.append(token)
                tok_cnt += 1

            segment_ids = [seg_id] * len(tokens)
            position_ids = list(range(len(tokens)))
            assert len(tokens) == len(label_ids) == len(position_ids) == len(
                segment_ids)

            tokens += [trainer.tokenizer.sep_token]
            label_ids += [-1]
            segment_ids += [seg_id]
            position_ids += [0]

            # The mask has 1 for real tokens and 0 for padding tokens.
            input_mask = [1] * len(tokens)

            # pad tensors
            pad_len = max_seq_length - len(tokens)
            tokens += [trainer.tokenizer.pad_token] * pad_len
            label_ids += [-1] * pad_len
            segment_ids += [0] * pad_len
            position_ids += [0] * pad_len
            input_mask += [0] * pad_len

            if trainer.args.lang_prefix:
                input_ids = trainer.tokenizer.convert_tokens_to_ids(tokens, language)
            else:
                input_ids = trainer.tokenizer.convert_tokens_to_ids(tokens)

            if ex_index < 3:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % tokens)
                logger.info("input_ids: %s" % input_ids)
                logger.info("segment_ids: %s" % segment_ids)
                logger.info("position_ids: %s" % position_ids)
                logger.info("input_mask: %s" % input_mask)
                logger.info("label: %s" %
                            all_labels[seq_span.start:seq_span.end])
                logger.info("label_ids: %s" % label_ids)

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              position_ids=position_ids,
                              label_ids=label_ids))
    # return features
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features],
                               dtype=torch.long)
    position_ids = torch.tensor([f.position_ids for f in features],
                                dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features],
                              dtype=torch.long)
    label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    data = TensorDataset(input_ids, segment_ids, position_ids, input_mask,
                         label_ids)
    if trainer.args.local_rank == -1:
        if evaluate:
            sampler = SequentialSampler(data)
        else:
            sampler = RandomSampler(data)
    else:
        sampler = DistributedSampler(data)
    dl = DataLoader(data, sampler=sampler, batch_size=batch_size)
    dl.lang_key = language
    return dl


class Trainer(BaseTrainer):
    def setup_args(self):
        super().setup_args()
        parser = self.parser
        parser.add_argument("--task_name",
                            default=None,
                            type=str,
                            required=True,
                            help="The name of the task to train.")
        EncoderConfigFactory.build_args(parser)
        DecoderConfigFactory.build_args(parser)

    def build_model(self):
        args = self.args

        assert self.processor is not None, 'call `setup_processor` first'
        num_labels = len(self.processor.get_labels())
        encoder_config = EncoderConfigFactory.build_config(args)
        decoder_config = DecoderConfigFactory.build_config(args)
        model_config = SeqLabelConfig(encoder_config=encoder_config,
                                      decoder_config=decoder_config,
                                      num_labels=num_labels)
        model = model_config.build()
        self.logger.info('Model config %r', model_config)
        self.logger.info('Model param %r', model)
        return model

    def setup_processor(self, lang):
        task_name = self.args.task_name.lower()

        processors = {
            "ner": processor.NERProcessor,
            "ner-wiki": processor.WikiNERProcessor,
            "pos": processor.POSProcessor,
            "pos-all": processor.RawPOSProcessor,
        }
        if task_name not in processors:
            raise ValueError(f"Task not found: {task_name}")

        self.processor = processors[task_name](lang)

    def setup_evaluator(self):
        task_name = self.args.task_name.lower()
        evaluators = {
            "ner": NEREvaluator(self.processor),
            "ner-wiki": NEREvaluator(self.processor),
            "pos": POSEvaluator(),
            "pos-all": POSEvaluator(),
        }
        self.evaluator = evaluators[task_name]
        assert self.model is not None
        self.model.set_evaluator(self.evaluator)

    def get_dataloader_from_example(self, examples, mode):
        assert mode in [Mode.train, Mode.test, Mode.dev]
        args = self.args
        batch_size = args.train_batch_size if mode == Mode.train else args.eval_batch_size
        evaluate = mode != Mode.train
        dataloader = examples_to_dataloader(self,
                                            examples,
                                            self.processor.language,
                                            args.max_seq_length,
                                            args.seq_stride,
                                            batch_size,
                                            evaluate=evaluate)
        return dataloader

    def model_selection_criterion(self, metrics):
        task_name = self.args.task_name.lower()
        if 'pos' in task_name:
            metric = metrics['acc']
        elif 'ner' in task_name:
            metric = metrics['f1']
        else:
            raise ValueError
        return metric

    def final_eval(self):
        args = self.args
        task_name = args.task_name.lower()
        save_fp = f'{args.output_dir}/model.pth'
        self.load_model(save_fp)

        for trg_lang in tqdm(args.trg_lang, desc="Eval Lang"):
            self.setup_processor(trg_lang)
            write_file = f'{args.output_dir}/eval/{args.lang}-{trg_lang}/eval_results.txt'
            maybe_mkdir(write_file)

            eval_loss, metrics = self.do_eval(Mode.test)
            self.write_eval(write_file, Mode.test, eval_loss, metrics)


if __name__ == "__main__":
    # place holder
    _ = torch.zeros(1).to('cuda')
    trainer = Trainer()
    if trainer.args.get_stat:
        import os
        from collections import Counter
        import pickle

        os.makedirs(os.path.dirname(trainer.args.get_stat), exist_ok=True)
        stats = dict()
        trainer.setup_processor(trainer.args.lang)
        for mode in [Mode.train, Mode.dev, Mode.test]:
            examples = trainer.get_examples(mode)
            if not examples:
                continue
            batch_size = trainer.args.train_batch_size if mode == Mode.train else trainer.args.eval_batch_size
            evaluate = mode != Mode.train
            data = examples_to_dataloader(trainer,
                                          examples,
                                          trainer.processor.language,
                                          trainer.args.max_seq_length,
                                          trainer.args.seq_stride,
                                          batch_size,
                                          evaluate,
                                          get_raw_data=True)
            vocab = Counter()
            for example, all_tokens, all_labels in data:
                vocab.update(all_tokens)
            stats[mode] = {}
            stats[mode]['nb_instance'] = len(data)
            stats[mode]['vocab'] = vocab

        with open(trainer.args.get_stat, 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        trainer.run()
