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
"""BERT finetuning runner."""

import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from pytorch_transformers import BertTokenizer, XLMTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import processor
from module import DecoderConfigFactory, EncoderConfigFactory, SeqClassConfig, SeqSepTopClassConfig
from module.evaluator import AccClassifierEvaluator, FullClassifierEvaluator
from trainer import BaseTrainer
from util import Mode, maybe_mkdir, truncate_seq_pair


@dataclass
class InputFeatures:
    """A single set of features of data."""
    input_ids: List[int]
    segment_ids: List[int]
    position_ids: List[int]
    input_mask: List[int]
    label_id: int
    ex_lang_id: int


def examples_to_dataloader(trainer: BaseTrainer, examples, language,
                           max_seq_length, batch_size, evaluate):
    """Loads a data file into a list of `InputBatch`s."""

    logger = trainer.logger
    label_map = {
        label: idx
        for (idx, label) in enumerate(trainer.processor.get_labels())
    }
    language_map = {
        language: idx
        for (idx, language) in enumerate(trainer.processor.get_languages())
    }

    if isinstance(trainer.tokenizer, XLMTokenizer):
        try:
            seg_id1 = trainer.tokenizer.lang2id[language]
            seg_id2 = trainer.tokenizer.lang2id[language]
        except:
            seg_id1, seg_id2 = 0, 0
    elif isinstance(trainer.tokenizer, BertTokenizer):
        seg_id1 = 0
        seg_id2 = 1
    else:
        raise ValueError(trainer.tokenizer)

    # Step 1: tokenize examples
    data = []
    for example in examples:
        tokens_a = trainer.tokenizer.tokenize(example.text_a)

        tokens_b = []
        if example.text_b:
            tokens_b = trainer.tokenizer.tokenize(example.text_b)

        data.append((example, tokens_a, tokens_b, example.label, example.lang))

    emperical_max_len = max([len(d[1]) + len(d[2]) for d in data]) + 2
    max_seq_length = min(max_seq_length, emperical_max_len)
    logger.info(f'Emperical seq len {max_seq_length}')

    # Step 2: create tensors
    features: List[InputFeatures] = []
    for (ex_index, d) in enumerate(data):
        example, tokens_a, tokens_b, label, ex_lang = d

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = [trainer.tokenizer.cls_token] + tokens_a
        segment_ids = [seg_id1] + [seg_id1] * len(tokens_a)
        position_ids = list(range(len(tokens)))
        assert len(tokens) == len(segment_ids) == len(position_ids)

        tokens += [trainer.tokenizer.sep_token]
        segment_ids += [seg_id1]
        position_ids += [0]

        if tokens_b:

            tokens += tokens_b
            segment_ids += [seg_id2] * len(tokens_b)
            position_ids += list(range(1, len(tokens_b) + 1))
            assert len(tokens) == len(segment_ids) == len(position_ids)

            tokens += [trainer.tokenizer.sep_token]
            segment_ids += [seg_id2]
            position_ids += [0]

        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(tokens)

        # pad tensors
        pad_len = max_seq_length - len(tokens)
        tokens += [trainer.tokenizer.pad_token] * pad_len
        segment_ids += [0] * pad_len
        position_ids += [0] * pad_len
        input_mask += [0] * pad_len

        input_ids = trainer.tokenizer.convert_tokens_to_ids(tokens)
        label_id = label_map[label]
        ex_lang_id = language_map[ex_lang]

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % tokens)
            logger.info("input_ids: %s" % input_ids)
            logger.info("segment_ids: %s" % segment_ids)
            logger.info("position_ids: %s" % position_ids)
            logger.info("input_mask: %s" % input_mask)
            logger.info("label: %s" % example.label)
            logger.info("label_id: %s" % label_id)
            logger.info("ex_lang_id: %s" % ex_lang_id)

        features.append(
            InputFeatures(input_ids=input_ids,
                          segment_ids=segment_ids,
                          position_ids=position_ids,
                          input_mask=input_mask,
                          label_id=label_id,
                          ex_lang_id=ex_lang_id))

    # Step 3: create dataloader with tensors
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features],
                               dtype=torch.long)
    position_ids = torch.tensor([f.position_ids for f in features],
                                dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features],
                              dtype=torch.long)
    label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    ex_lang_ids = torch.tensor([f.ex_lang_id for f in features], dtype=torch.long)
    data = TensorDataset(input_ids, segment_ids, position_ids, input_mask,
                         label_ids, ex_lang_ids)
    if trainer.args.local_rank == -1:
        if evaluate:
            sampler = SequentialSampler(data)
        else:
            sampler = RandomSampler(data)
    else:
        sampler = DistributedSampler(data)
    dl = DataLoader(data, sampler=sampler, batch_size=batch_size)
    """ex_lang_id is the language in the current batch, whereas
    lang_key is the hyperparameter specifying which language(s) to use
    from the training set."""
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
        if args.separate_top:
            model_config = SeqSepTopClassConfig(encoder_config=encoder_config,
                                                decoder_config=decoder_config,
                                                num_labels=num_labels)
        else:
            model_config = SeqClassConfig(encoder_config=encoder_config,
                                          decoder_config=decoder_config,
                                          num_labels=num_labels)
        model = model_config.build()
        self.logger.info('Model config %r', model_config)
        self.logger.info('Model param %r', model)
        return model

    def setup_processor(self, lang=None):
        task_name = self.args.task_name.lower()

        processors = {
            "cola": processor.ColaProcessor,
            "mnli": processor.MnliProcessor,
            "mrpc": processor.MrpcProcessor,
            "xnli": processor.XnliProcessor,
            "mldoc": processor.MLDocProcessor,
            "tobacco": processor.TobaccoProcessor,
            "tobacco-feedback": processor.TobaccoFeedbackProcessor,
            "langid": processor.LangIDProcessor,
        }
        if task_name not in processors:
            raise ValueError(f"Task not found: {task_name}")

        if task_name == 'xnli' or task_name == 'mldoc' or 'tobacco' in task_name:
            self.processor = processors[task_name](lang)
        elif task_name == 'langid':
            self.processor = processors[task_name](self.args.data_dir)
        else:
            self.processor = processors[task_name]()

    def setup_evaluator(self):
        task_name = self.args.task_name.lower()
        if 'tobacco' in task_name:
            evaluator = FullClassifierEvaluator()
        elif task_name == 'langid':
            evaluator = FullClassifierEvaluator(average='macro')
        else:
            evaluator = AccClassifierEvaluator()
        self.evaluator = evaluator
        assert self.model is not None
        self.model.set_evaluator(self.evaluator)

    def get_dataloader_from_example(self, examples, mode):
        assert mode in [Mode.train, Mode.test, Mode.dev]
        args = self.args
        if args.separate_top:
            batch_size = 1
        else:
            batch_size = args.train_batch_size if mode == Mode.train else args.eval_batch_size
        evaluate = mode != Mode.train
        dataloader = examples_to_dataloader(self,
                                            examples,
                                            self.processor.language,
                                            args.max_seq_length,
                                            batch_size,
                                            evaluate=evaluate)
        return dataloader

    def model_selection_criterion(self, metrics):
        task_name = self.args.task_name.lower()
        if 'tobacco' in task_name:
            metric = metrics['f1']
        else:
            metric = metrics['acc']
        return metric

    def final_eval(self):
        args = self.args
        task_name = args.task_name.lower()
        save_fp = f'{args.output_dir}/model.pth'
        self.load_model(save_fp)

        if task_name == 'xnli' or task_name == 'mldoc' or 'tobacco' in task_name:
            for trg_lang in tqdm(args.trg_lang, desc="Eval Lang"):
                self.setup_processor(trg_lang)
                write_file = f'{args.output_dir}/eval/{args.lang}-{trg_lang}/eval_results.txt'
                maybe_mkdir(write_file)

                eval_loss, metrics = self.do_eval(Mode.test)
                self.write_eval(write_file, Mode.test, eval_loss, metrics)
        else:
            for mode in [Mode.dev, Mode.test]:
                write_file = f'{args.output_dir}/eval_results.txt'
                maybe_mkdir(write_file)

                eval_loss, metrics = self.do_eval(mode)
                self.write_eval(write_file, mode, eval_loss, metrics)


if __name__ == "__main__":
    # place holder
    _ = torch.zeros(1).to('cuda')
    trainer = Trainer()
    trainer.run()
