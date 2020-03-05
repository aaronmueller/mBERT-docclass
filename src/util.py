import logging
import os
import re
import sys
from enum import Enum
from typing import Dict

import numpy as np
import torch
from sklearn import metrics

XLM_LANG2ID = dict()
XLM_LANG2ID["ar"] = 0
XLM_LANG2ID["bg"] = 1
XLM_LANG2ID["de"] = 2
XLM_LANG2ID["el"] = 3
XLM_LANG2ID["en"] = 4
XLM_LANG2ID["es"] = 5
XLM_LANG2ID["fr"] = 6
XLM_LANG2ID["hi"] = 7
XLM_LANG2ID["ru"] = 8
XLM_LANG2ID["sw"] = 9
XLM_LANG2ID["th"] = 10
XLM_LANG2ID["tr"] = 11
XLM_LANG2ID["ur"] = 12
XLM_LANG2ID["vi"] = 13
XLM_LANG2ID["zh"] = 14
#########################
XLM_LANG2ID["UD_Arabic-PADT"] = 0
XLM_LANG2ID["UD_Bulgarian-BTB"] = 1
XLM_LANG2ID["UD_German-GSD"] = 2
XLM_LANG2ID["UD_Greek-GDT"] = 3
XLM_LANG2ID["UD_English-EWT"] = 4
XLM_LANG2ID["UD_Spanish-GSD"] = 5
XLM_LANG2ID["UD_French-GSD"] = 6
XLM_LANG2ID["UD_Hindi-HDTB"] = 7
XLM_LANG2ID["UD_Russian-GSD"] = 8
XLM_LANG2ID["UD_Thai-PUD"] = 10
XLM_LANG2ID["UD_Turkish-IMST"] = 11
XLM_LANG2ID["UD_Urdu-UDTB"] = 12
XLM_LANG2ID["UD_Vietnamese-VTB"] = 13
XLM_LANG2ID["UD_Chinese-GSD"] = 14


def maybe_mkdir(filename):
    '''
    maybe mkdir
    '''
    path = os.path.dirname(filename)
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def get_logger(log_file, log_level='info'):
    '''
    create logger and output to file and stdout
    '''
    assert log_level in ['info', 'debug']
    fmt = '%(asctime)s - %(levelname)s -   %(message)s'
    datefmt = '%m/%d/%Y %H:%M:%S'
    logger = logging.getLogger()
    logger.handlers = []
    log_level = {'info': logging.INFO, 'debug': logging.DEBUG}[log_level]
    logger.setLevel(log_level)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(stream)
    filep = logging.FileHandler(log_file, mode='a')
    filep.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(filep)
    return logger


class NamedEnum(Enum):
    def __str__(self):
        return self.value


class Mode(NamedEnum):
    train = 'train'
    dev = 'dev'
    test = 'test'


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()