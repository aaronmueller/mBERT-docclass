import argparse
import logging
import os
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from pytorch_transformers import AdamW, BertTokenizer, WarmupLinearSchedule, XLMTokenizer
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from module.tokenization import NewXLMTokenizer, LangPrefixXLMTokenizer
from util import Mode, get_logger, maybe_mkdir


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class BaseTrainer(object):
    def __init__(self):
        self.device = None
        self.n_gpu = None
        self.model = None
        self.processor = None
        self.evaluator = None
        self.parser = argparse.ArgumentParser()
        self.setup_args()
        self.args = self.parser.parse_args()
        self.logger = None
        self.setup_misc()
        if self.args.encoder == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_mode)
        elif self.args.encoder == 'xlm':
            self.logger.warn(
                'WE ASSUME THE TEXT IS PRETOKENIZED AND ONLY DO THE FOLLOWING:'
            )
            self.logger.warn(
                '1. Remove invalid character and clean up white space')
            self.logger.warn(
                '2. Lower case, renormalize unicode and strip accents')
            self.logger.warn(
                '3. Split text by white space and do BPE on each "word"')
            if self.args.diff_bpe:
                if self.args.lang_prefix:
                    self.tokenizer = LangPrefixXLMTokenizer.from_pretrained(
                        self.args.xlm_mode,
                        do_lowercase_and_remove_accent=self.args.do_lower_case)
                else:
                    self.tokenizer = NewXLMTokenizer.from_pretrained(
                        self.args.xlm_mode,
                        do_lowercase_and_remove_accent=self.args.do_lower_case)
            else:
                self.tokenizer = XLMTokenizer.from_pretrained(
                    self.args.xlm_mode,
                    do_lowercase_and_remove_accent=self.args.do_lower_case)
        else:
            raise ValueError(self.args.encoder)
        self.param_optimizer, self.optimizer, self.scheduler = None, None, None
        self.global_step, self.total_step = 0, 0
        self.num_train_steps, self.warmup_linear = None, None
        self.last_eval, self.best_eval = 0, float('-inf')

    def setup_args(self):
        parser = self.parser
        ## Required parameters
        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help=
            "The input data dir. Should contain the .tsv files (or other data files) for the task."
        )
        parser.add_argument("--lang",
                            default="en",
                            type=str,
                            help="Run language")
        parser.add_argument("--trg_lang",
                            default="",
                            type=str,
                            help="Eval of target language")
        parser.add_argument(
            "--output_dir",
            default=None,
            type=str,
            required=True,
            help=
            "The output directory where the model checkpoints will be written."
        )

        ## Other parameters
        parser.add_argument('--schedule_half',
                            default=False,
                            action='store_true',
                            help="Half scheduler")
        parser.add_argument("--min_lr",
                            default=1e-5,
                            type=float,
                            help="The minimum learning rate of half scheduler")
        parser.add_argument("--load",
                            default="",
                            type=str,
                            help="Load trained model")
        parser.add_argument('--do_lower_case',
                            default=False,
                            action='store_true',
                            help="Lower case in tokenization")
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help=
            "The maximum total input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.")
        parser.add_argument(
            "--seq_stride",
            default=64,
            type=int,
            help=
            "When splitting up a long sequence into chunks, how much stride to take between chunks."
        )
        parser.add_argument("--do_train",
                            default=False,
                            action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--lang_prefix",
                            default=False,
                            action='store_true',
                            help="Try language prefix for indexing vocab.")
        parser.add_argument("--diff_bpe",
                            default=False,
                            action='store_true',
                            help="Run different BPE for each languages")
        parser.add_argument("--do_eval",
                            default=False,
                            action='store_true',
                            help="Whether to run eval on the dev set.")
        parser.add_argument("--no_eval_dev",
                            default=False,
                            action='store_true',
                            help="Whether not to run eval on the dev set.")
        parser.add_argument("--no_eval_test",
                            default=False,
                            action='store_true',
                            help="Whether not to run eval on the test set.")
        parser.add_argument("--train_batch_size",
                            default=32,
                            type=int,
                            help="Per GPU batch size for training.")
        parser.add_argument("--eval_batch_size",
                            default=32,
                            type=int,
                            help="Per GPU batch size for eval.")
        parser.add_argument("--learning_rate",
                            default=5e-5,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay",
                            default=0.01,
                            type=float,
                            help="Weight deay if we apply some.")
        parser.add_argument("--adam_epsilon",
                            default=1e-8,
                            type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm",
                            default=1.0,
                            type=float,
                            help="Max gradient norm.")
        parser.add_argument("--num_train_epochs",
                            default=3.0,
                            type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help=
            "Proportion of training to perform linear learning rate warmup for. "
            "E.g., 0.1 = 10%% of training.")
        parser.add_argument("--separate_top",
                            default=False,
                            action='store_true',
                            help="Enables language-specific top-level classification layers.")
        parser.add_argument("--no_cuda",
                            default=False,
                            action='store_true',
                            help="Whether not to use CUDA when available")
        parser.add_argument("--local_rank",
                            type=int,
                            default=-1,
                            help="local_rank for distributed training on gpus")
        parser.add_argument('--seed',
                            type=int,
                            default=42,
                            help="random seed for initialization")
        parser.add_argument(
            '--gradient_accumulation_steps',
            type=int,
            default=1,
            help=
            "Number of updates steps to accumualte before performing a backward/update pass."
        )
        parser.add_argument(
            '--fp16',
            action='store_true',
            help=
            "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
        )
        parser.add_argument(
            '--fp16_opt_level',
            type=str,
            default='O1',
            help=
            "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            "See details at https://nvidia.github.io/apex/amp.html")
        parser.add_argument('--get_stat',
                            default='',
                            type=str,
                            help="Get stat instead")

    def setup_misc(self):
        args = self.args

        if args.trg_lang:
            if os.path.isfile(args.trg_lang):
                args.trg_lang = [
                    l.strip() for l in open(args.trg_lang, 'r').readlines()
                ]
            else:
                args.trg_lang = args.trg_lang.split(',')
        else:
            args.trg_lang = args.lang

        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            print(
                f"Output directory ({args.output_dir}) already exists and is not empty."
            )
        os.makedirs(args.output_dir, exist_ok=True)

        logger = get_logger(f'{self.args.output_dir}/exp.log')
        self.logger = logger
        for key, value in vars(args).items():
            logger.info('command line argument: %s - %r', key, value)

        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available()
                                  and not args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')

        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available()
                                  and not args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            n_gpu = 1
        logger.info(
            f"device: {device}, n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
        )
        self.device, self.n_gpu = device, n_gpu

        if args.gradient_accumulation_steps < 1:
            raise ValueError(
                "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
                .format(args.gradient_accumulation_steps))

        args.train_batch_size = int(args.train_batch_size * self.n_gpu)
        args.eval_batch_size = int(args.eval_batch_size * self.n_gpu)

        set_seed(args.seed, self.n_gpu)

        if not args.do_train and not args.do_eval:
            raise ValueError(
                "At least one of `do_train` or `do_eval` must be True.")

    def setup_model(self):
        args = self.args
        model = self.build_model()
        assert model is not None
        if args.fp16:
            model.half()
        model.to(self.device)
        # Distributed and parallel training
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True)
        elif self.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        self.model = model

    def build_model(self):
        raise NotImplementedError

    def setup_processor(self, lang=None):
        raise NotImplementedError

    def setup_evaluator(self):
        raise NotImplementedError

    def load_model(self, model_path, load_direct=False):
        args, model = self.args, self.model
        assert os.path.isfile(model_path)

        if load_direct:
            m = torch.load(model_path)
            model.load_state_dict(m.state_dict())
        else:
            model.load_state_dict(torch.load(model_path))
        if args.fp16:
            model.half()
        else:
            model.float()
        model.to(self.device)
        if args.local_rank != -1:
            self.model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank)
        elif self.n_gpu > 1:
            self.model = nn.DataParallel(model)

    def setup_optim(self, t_total, warmup_step):
        args, model = self.args, self.model

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters.append({
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            args.weight_decay
        })
        optimizer_grouped_parameters.append({
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.0
        })
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer,
                                         warmup_steps=warmup_step,
                                         t_total=t_total)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            self.model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.fp16_opt_level)
        if args.schedule_half:
            scheduler = ReduceLROnPlateau(optimizer,
                                          factor=0.5,
                                          patience=0,
                                          min_lr=args.min_lr,
                                          mode='max')
        self.scheduler, self.optimizer = scheduler, optimizer

    def get_examples(self, mode):
        return self.processor.get_examples(self.args.data_dir, mode)

    def train(self, dataloader, epoch_idx):
        args, model = self.args, self.model
        scheduler, optimizer = self.scheduler, self.optimizer
        model.train()
        running_loss, step = 0, 0
        lr = optimizer.param_groups[0]['lr']
        self.logger.info(f'lr = {lr} at the begin of epoch {epoch_idx}')
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            batch = tuple(t.to(self.device) for t in batch)
            if args.separate_top:
                loss = model(*batch, lang_key=dataloader.lang_key)
            else:
                loss = model(*batch, lang_key=dataloader.lang_key)

            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                               args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)

            running_loss += loss.item()
            step += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if not args.schedule_half:
                    scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                self.global_step += 1
        return running_loss / step

    def evaluate(self, dataloader):
        model = self.model
        model.eval()
        model.reset_metrics()
        #if args.separate_eval:
        #    eval_loss = defaultdict(int)
        #else:
        eval_loss = 0
        step = 0
        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                loss = model(*batch, lang_key=dataloader.lang_key)
                #if args.separate_eval:
                #    ex_lang = batch[-1].item()
                #    eval_loss[ex_lang] += loss.mean().item()
                #else:
                eval_loss += loss.mean().item()
            step += 1

        #if args.separate_eval:
        #    for ex_lang in eval_loss.keys():
        #        eval_loss[ex_lang] = eval_loss[ex_lang] / step
        #    return eval_loss, model.get_metrics(True)
        #else:
        return eval_loss / step, model.get_metrics(True)

    def update_lr_save_model_maybe_stop(self, eval_res, epoch_idx):
        stop_early = True
        if self.args.schedule_half:
            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(eval_res)
            curr_lr = self.optimizer.param_groups[0]['lr']

            if self.last_eval >= eval_res and prev_lr == curr_lr == self.args.min_lr:
                self.logger.info(
                    f'Early stopping triggered with epoch {epoch_idx} (previous: {self.last_eval}, current: {eval_res})'
                )
                return stop_early
            self.last_eval = eval_res

        if eval_res >= self.best_eval:
            self.best_eval = eval_res
            save_fp = os.path.join(self.args.output_dir, "model.pth")
            torch.save(self.model.state_dict(), save_fp)
        return not stop_early

    def write_eval(self, filepath, mode, eval_loss, metrics, running_loss=0):
        result = {
            'eval_loss': eval_loss,
            'global_step': self.global_step,
            'running_loss': running_loss
        }
        for tag, metric in metrics.items():
            result[f'{mode}-{tag}'] = metric

        with open(filepath, "a") as writer:
            self.logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                self.logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    def get_dataloader_from_example(self, examples, mode):
        # assert mode in ['train', 'eval']
        raise NotImplementedError

    def model_selection_criterion(self, metrics):
        raise NotImplementedError

    def do_train(self):
        args, logger = self.args, self.logger
        train_examples = self.get_examples(Mode.train)
        if not train_examples:
            logger.warn('Empty training file')
            raise ValueError('Empty training file')
        train_dataloader = self.get_dataloader_from_example(
            train_examples, Mode.train)

        t_total = len(train_dataloader) // args.gradient_accumulation_steps
        t_total *= args.num_train_epochs
        warmup_step = int(t_total * args.warmup_proportion)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d",
                    len(train_dataloader) * args.train_batch_size)
        logger.info("  Num batchs = %d", len(train_dataloader))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Batch size (on %d GPU) = %d, batch size per GPU = %d",
                    self.n_gpu, args.train_batch_size,
                    args.train_batch_size // self.n_gpu)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size * args.gradient_accumulation_steps *
            (torch.distributed.get_world_size()
             if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d",
                    args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Total warmup steps = %d", warmup_step)

        # Prepare optimizer
        self.setup_optim(t_total, warmup_step)

        for epoch_idx in trange(int(args.num_train_epochs), desc="Epoch"):
            running_loss = self.train(train_dataloader, epoch_idx)

            try:
                eval_loss, metrics = self.do_eval(Mode.dev)
            except Exception as e:
                logger.error(traceback.format_exc())
                save_fp = os.path.join(args.output_dir, "model.pth")
                torch.save(self.model.state_dict(), save_fp)
                continue

            write_file = f'{args.output_dir}/eval_results.txt'
            self.write_eval(write_file, Mode.dev, eval_loss, metrics,
                            running_loss)

            metric = self.model_selection_criterion(metrics)
            if self.update_lr_save_model_maybe_stop(metric, epoch_idx):
                # stop early
                break

    def final_eval(self):
        raise NotImplementedError
    
    # LOOK INTO THIS: LANGUAGE-SPECIFIC EVAL
    def do_eval(self, mode):
        args, logger = self.args, self.logger
        examples = self.get_examples(mode)
        if not examples:
            logger.warn(f'Empty {mode} file')
            raise ValueError(f'Empty {mode} file')
        dataloader = self.get_dataloader_from_example(examples, Mode.test)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_loss, metrics = self.evaluate(dataloader)
        return eval_loss, metrics

    def run(self):
        args = self.args

        self.setup_processor(args.lang)
        self.setup_model()
        self.setup_evaluator()

        if args.do_train:
            try:
                self.do_train()
                self.final_eval()
            except Exception as e:
                self.logger.error(traceback.format_exc())

        if args.do_eval and not args.do_train and args.local_rank in [-1, 0]:
            self.load_model(args.load)

            modes = []
            if not args.no_eval_dev:
                modes.append(Mode.dev)
            if not args.no_eval_test:
                modes.append(Mode.test)

            for trg_lang in tqdm(args.trg_lang, desc="Eval Lang"):
                self.setup_processor(trg_lang)
                write_file = f'{args.output_dir}/eval/{args.lang}-{trg_lang}/eval_results.txt'
                maybe_mkdir(write_file)

                for mode in modes:
                    eval_loss, metrics = self.do_eval(mode)
                    self.write_eval(write_file, mode, eval_loss, metrics)
