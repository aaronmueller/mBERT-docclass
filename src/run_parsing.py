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
"""BERT finetuning parsing."""

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
from module import DecoderConfigFactory, EncoderConfigFactory, ParsingConfig
from module.evaluator import ParsingEvaluator
from trainer import BaseTrainer
from util import Mode, maybe_mkdir


@dataclass
class InputFeatures:
    """A single set of features of data."""
    input_ids: List[int]
    pos_ids: List[int]
    segment_ids: List[int]
    position_ids: List[int]
    input_mask: List[int]
    nonword_mask: List[int]
    head_ids: List[int]
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
                           max_sent_size=256,
                           get_raw_data=False):
    """Loads a data file into a list of `InputBatch`s."""
    logger = trainer.logger
    label_map = {
        label: idx
        for (idx, label) in enumerate(trainer.processor.get_labels())
    }
    label_map[-1] = -1  # pad label
    pos_map = {
        pos: idx
        for (idx, pos) in enumerate(trainer.processor.get_pos())
    }

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
        all_heads = []
        all_pos_tags = []
        all_labels = []
        word_pos_to_wp_pos = {}
        word_pos_to_wp_pos[0] = 0
        word_pos_to_wp_pos[-1] = -1
        assert len(example.text) == len(example.pos) == \
               len(example.head) == len(example.label)
        for (i, (token, pos, head, label)) in enumerate(
                zip(example.text, example.pos, example.head, example.label)):
            if isinstance(trainer.tokenizer, XLMTokenizer):
                sub_tokens = trainer.tokenizer.tokenize(token, lang=language)
            elif isinstance(trainer.tokenizer, BertTokenizer):
                sub_tokens = trainer.tokenizer.tokenize(token)
            else:
                raise ValueError(trainer.tokenizer)
            if len(sub_tokens) >= 1:
                # Account for CLS
                word_pos_to_wp_pos[i + 1] = len(all_tokens) + 1
                all_tokens.append(sub_tokens[0])
                all_pos_tags.append(pos)
                all_heads.append(head)
                all_labels.append(label)
            if len(sub_tokens) > 1:
                for sub_token in sub_tokens[1:]:
                    all_tokens.append(sub_token)
                    all_pos_tags.append(pos)
                    all_heads.append(-1)
                    all_labels.append(-1)
        data.append((example, all_tokens, all_pos_tags, all_heads, all_labels,
                     word_pos_to_wp_pos))
    if get_raw_data:
        return data

    emperical_max_len = max([len(d[1]) for d in data]) + 2
    if evaluate:
        max_seq_length = min(max_sent_size, emperical_max_len)
        logger.info(f'Eval max seq length {max_seq_length}')
    else:
        max_seq_length = min(max_seq_length, emperical_max_len)
    logger.info(f'Eval {evaluate} with seq len {max_seq_length}')

    # Step 2: create tensors
    features: List[InputFeatures] = []
    skip_cnt = 0
    for (ex_index, d) in enumerate(data):
        example, all_tokens, all_pos_tags, all_heads, all_labels, word_pos_to_wp_pos = d

        # We can have sequences that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `seq_stride`.
        seq_spans = []
        # Account for [CLS] and [SEP] with "- 2"
        max_tokens_for_seq = max_seq_length - 2
        start_offset = 0
        if len(all_tokens) > max_tokens_for_seq:
            skip_cnt += 1
            logger.info(
                f'Skip sent {skip_cnt} with {len(all_tokens)} word piece')
            continue
        while start_offset < len(all_tokens):
            length = len(all_tokens) - start_offset
            if length > max_tokens_for_seq:
                length = max_tokens_for_seq
            seq_spans.append(
                SeqSpan(start=start_offset, end=start_offset + length))
            if start_offset + length == len(all_tokens):
                break
            start_offset += min(length, seq_stride)

        # FIXME: Only use the first chunck for training
        seq_spans = [seq_spans[0]]
        for span in seq_spans:
            tokens: List[str] = [trainer.tokenizer.cls_token]
            pos_ids: List[str] = [pos_map["_"]]
            heads: List[int] = [-1]
            label_ids: List[int] = [-1]
            nonword_mask: List[int] = [0]

            for token, pos, head, label in zip(
                    all_tokens[span.start:span.end],
                    all_pos_tags[span.start:span.end],
                    all_heads[span.start:span.end],
                    all_labels[span.start:span.end]):
                tokens.append(token)
                pos_ids.append(pos_map[pos])
                if head == -1:
                    nonword_mask.append(0)
                else:
                    nonword_mask.append(1)
                head = word_pos_to_wp_pos[head]
                if head >= max_seq_length:
                    head = -1
                    label = -1
                heads.append(head)
                if label in label_map:
                    label_ids.append(label_map[label])
                else:
                    label_ids.append(label_map[processor.UNK])

            segment_ids = [seg_id] * len(tokens)
            position_ids = list(range(len(tokens)))
            assert len(tokens) == len(pos_ids) == len(heads) == len(label_ids)
            assert len(label_ids) == len(nonword_mask) == len(segment_ids)
            assert len(segment_ids) == len(position_ids)

            tokens += [trainer.tokenizer.sep_token]
            pos_ids += [pos_map["_"]]
            heads += [-1]
            label_ids += [-1]
            nonword_mask += [0]
            segment_ids += [seg_id]
            position_ids += [0]

            # The mask has 1 for real tokens and 0 for padding tokens.
            input_mask = [1] * len(tokens)

            # pad tensors
            pad_len = max_seq_length - len(tokens)
            tokens += [trainer.tokenizer.pad_token] * pad_len
            pos_ids += [pos_map["_"]] * pad_len
            heads += [-1] * pad_len
            label_ids += [-1] * pad_len
            nonword_mask += [0] * pad_len
            segment_ids += [0] * pad_len
            position_ids += [0] * pad_len
            input_mask += [0] * pad_len

            if trainer.args.lang_prefix:
                input_ids = trainer.tokenizer.convert_tokens_to_ids(tokens, language)
            else:
                input_ids = trainer.tokenizer.convert_tokens_to_ids(tokens)

            if ex_index < 3:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % tokens)
                logger.info("input_ids: %s" % input_ids)
                logger.info("pos_tags: %s" % example.pos)
                logger.info("pos_ids: %s" % pos_ids)
                logger.info("segment_ids: %s" % segment_ids)
                logger.info("position_ids: %s" % position_ids)
                logger.info("input_mask: %s" % input_mask)
                logger.info("nonword_mask: %s" % nonword_mask)
                logger.info("head raw: %s" % example.head)
                logger.info("head: %s" % heads)
                logger.info("label: %s" % example.label)
                logger.info("label_ids: %s" % label_ids)

            features.append(
                InputFeatures(input_ids=input_ids,
                              pos_ids=pos_ids,
                              input_mask=input_mask,
                              nonword_mask=nonword_mask,
                              segment_ids=segment_ids,
                              position_ids=position_ids,
                              head_ids=heads,
                              label_ids=label_ids))
    # return features
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features],
                              dtype=torch.long)
    nonword_mask = torch.tensor([f.nonword_mask for f in features],
                                dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features],
                               dtype=torch.long)
    position_ids = torch.tensor([f.position_ids for f in features],
                               dtype=torch.long)
    head_ids = torch.tensor([f.head_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    data = TensorDataset(input_ids, pos_ids, segment_ids, position_ids, input_mask,
                         nonword_mask, label_ids, head_ids)
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
                            default='parsing',
                            type=str,
                            help="The name of the task to train.")
        parser.add_argument('--use_pos',
                            default=False,
                            action='store_true',
                            help="Use POS tag as part of the input")
        EncoderConfigFactory.build_args(parser)
        DecoderConfigFactory.build_args(parser)

    def build_model(self):
        args = self.args

        assert self.processor is not None, 'call `setup_processor` first'
        num_labels = len(self.processor.get_labels())
        num_pos = len(self.processor.get_pos())
        encoder_config = EncoderConfigFactory.build_config(args)
        decoder_config = DecoderConfigFactory.build_config(args)
        model_config = ParsingConfig(encoder_config=encoder_config,
                                     decoder_config=decoder_config,
                                     num_labels=num_labels,
                                     num_pos=num_pos,
                                     use_pos=args.use_pos)
        model = model_config.build()
        self.logger.info('Model config %r', model_config)
        self.logger.info('Model param %r', model)
        return model

    def setup_processor(self, lang):
        task_name = self.args.task_name.lower()

        processors = {
            "parsing": processor.ParsingProcessor,
            "parsing-all": processor.RawParsingProcessor,
        }
        if task_name not in processors:
            raise ValueError(f"Task not found: {task_name}")

        self.processor = processors[task_name](lang)

    def setup_evaluator(self):
        self.evaluator = ParsingEvaluator()
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
        metric = metrics['LAS']
        return metric

    def final_eval(self):
        args = self.args
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
        stats = {}
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
            for example, all_tokens, all_pos_tags, all_heads, all_labels, word_pos_to_wp_pos in data:
                vocab.update(all_tokens)
            stats[mode] = {}
            stats[mode]['nb_instance'] = len(data)
            stats[mode]['vocab'] = vocab

        with open(trainer.args.get_stat, 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        trainer.run()
