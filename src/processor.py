import os
import csv
import glob
import json

import langcodes
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize

from util import Mode

UNK = '<UNK>'
POS_TO_IGNORE = {'``', "''", ':', ',', '.', 'PU', 'PUNCT', 'SYM'}


def sent_tokenize(text, lang='en'):
    lang = langcodes.Language(lang).language_name().lower()
    try:
        return nltk_sent_tokenize(text, language=lang)
    except:
        return nltk_sent_tokenize(text)


class BaseProcessor(object):
    """Base class for data converters for data sets."""

    def __init__(self):
        self.cache = dict()

    def get_train_examples(self, data_dir):
        """Gets a collection of `LabelExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `LabelExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `LabelExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_examples(self, data_dir, mode):
        if (data_dir, mode) not in self.cache:
            if mode == Mode.train:
                examples = self.get_train_examples(data_dir)
            elif mode == Mode.dev:
                examples = self.get_dev_examples(data_dir)
            elif mode == Mode.test:
                examples = self.get_test_examples(data_dir)
            else:
                raise ValueError('Wrong mode', mode)
            self.cache[(data_dir, mode)] = examples
        return self.cache[(data_dir, mode)]


##################################################################
# CLASSIFICATION
##################################################################


class ClassificationExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, language=None):
        """Constructs a ClassificationExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.lang = language
        self.batch_example = None


class TSVProcessor(BaseProcessor):
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class XnliProcessor(TSVProcessor):
    """Processor for the XNLI data set."""

    def __init__(self, lang):
        super().__init__()
        self.language = lang

    def get_train_examples(self, data_dir):
        fp = f'{data_dir}/multinli/multinli.train.{self.language}.tsv'
        examples = []
        for (i, line) in enumerate(self._read_tsv(fp)):
            if i == 0:
                continue
            guid = "train-%d" % (i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            if label == "contradictory":
                label = "contradiction"
            examples.append(
                ClassificationExample(guid=guid,
                                      text_a=text_a,
                                      text_b=text_b,
                                      label=label))
        return examples

    def get_dev_examples(self, data_dir):
        fp = f'{data_dir}/xnli.dev.tsv'
        examples = []
        for (i, line) in enumerate(self._read_tsv(fp)):
            if i == 0:
                continue
            guid = "dev-%d" % (i)
            language = line[0]
            if language != self.language:
                continue
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            examples.append(
                ClassificationExample(guid=guid,
                                      text_a=text_a,
                                      text_b=text_b,
                                      label=label))
        return examples

    def get_test_examples(self, data_dir):
        fp = f'{data_dir}/xnli.test.tsv'
        examples = []
        for (i, line) in enumerate(self._read_tsv(fp)):
            if i == 0:
                continue
            guid = "test-%d" % (i)
            language = line[0]
            if language != self.language:
                continue
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            examples.append(
                ClassificationExample(guid=guid,
                                      text_a=text_a,
                                      text_b=text_b,
                                      label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]


class MLDocProcessor(TSVProcessor):
    """Processor for the MLDoc data set."""

    def __init__(self, lang):
        super().__init__()
        self.language = lang

    def get_train_examples(self, data_dir):
        fp = f"{data_dir}/{self.language}.train.1000"
        return self._create_examples(self._read_tsv(fp), "train")

    def get_dev_examples(self, data_dir):
        fp = f"{data_dir}/{self.language}.dev"
        return self._create_examples(self._read_tsv(fp), "dev")

    def get_languages(self):
        return [""]

    def get_test_examples(self, data_dir):
        fp = f"{data_dir}/{self.language}.test"
        return self._create_examples(self._read_tsv(fp), "test")

    def get_labels(self):
        """See base class."""
        return ["CCAT", "ECAT", "GCAT", "MCAT"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            sentences = sent_tokenize(line[1], lang=self.language)
            text_a = sentences[0]
            if len(sentences) > 1:
                text_b = sentences[1]
            else:
                text_b = None
            label = line[0]
            examples.append(
                ClassificationExample(guid=guid,
                                      text_a=text_a,
                                      text_b=text_b,
                                      label=label))
        return examples


class LangIDProcessor(MLDocProcessor):
    def __init__(self, data_dir):
        super().__init__()
        self.labels = [
            lang.strip()
            for lang in open(f"{data_dir}/observed.lang.txt", "r").readlines()
        ]

    def get_train_examples(self, data_dir):
        fp = f"{data_dir}/langid.train"
        return self._create_examples(self._read_tsv(fp), "train")

    def get_dev_examples(self, data_dir):
        fp = f"{data_dir}/langid.dev"
        return self._create_examples(self._read_tsv(fp), "dev")

    def get_test_examples(self, data_dir):
        fp = f"{data_dir}/langid.test"
        return self._create_examples(self._read_tsv(fp), "test")

    def get_labels(self):
        return self.labels


class TobaccoProcessor(BaseProcessor):
    """Processor for the Tobacco Watcher data set."""

    def __init__(self, lang):
        super().__init__()
        self.language = lang

    def get_train_examples(self, data_dir):
        fp = f"{data_dir}/20180723.translated-to-{self.language}"
        return self._create_examples(self._read_jsonl(fp), "train")

    def get_dev_examples(self, data_dir):
        fp = f"{data_dir}/20180723.translated-to-{self.language}"
        return self._create_examples(self._read_jsonl(fp), "dev")

    def get_test_examples(self, data_dir):
        fp = f"{data_dir}/20180723.translated-to-{self.language}"
        return self._create_examples(self._read_jsonl(fp), "test")

    def get_labels(self):
        return ["no", "yes"]  # about_tobacco

    def _read_jsonl(self, input_file):
        with open(input_file, 'r') as fp:
            objs = []
            for line in fp.readlines():
                objs.append(json.loads(line))
        return objs

    def _create_examples(self, objs, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, obj) in enumerate(objs):
            if obj['fold'] != set_type:
                continue
            guid = "%s-%s" % (set_type, i)
            # text_a = obj['title']
            sentences = sent_tokenize(obj['body'], lang=self.language)
            if len(sentences) >= 2:
                text_a, text_b = obj['title'] + sentences[0], sentences[1]
            elif len(sentences) == 1:
                text_a, text_b = obj['title'] + sentences[0], None
            else:
                text_a, text_b = obj['title'], None
            label = obj['label']['about_tobacco']
            examples.append(
                ClassificationExample(guid=guid,
                                      text_a=text_a,
                                      text_b=text_b,
                                      label=label))
        return examples


class TobaccoFeedbackProcessor(TobaccoProcessor):
    """Processor for the Tobacco Watcher data set."""

    def get_train_examples(self, data_dir):
        if os.path.isfile(data_dir):
            fp = data_dir
        else:
            fp = f"{data_dir}/clean_feedback.json"
        return self._create_examples(self._read_jsonl(fp), "train")

    def get_dev_examples(self, data_dir):
        if os.path.isfile(data_dir):
            fp = data_dir
        else:
            fp = f"{data_dir}/clean_feedback.json"
        return self._create_examples(self._read_jsonl(fp), "dev")

    def get_test_examples(self, data_dir):
        if os.path.isfile(data_dir):
            fp = data_dir
        else:
            fp = f"{data_dir}/clean_feedback.json"
        return self._create_examples(self._read_jsonl(fp), "test")
    
    def get_languages(self):
        """Returns list of all languages in training set."""
        return ["ar", "bn", "de", "en", "es", "fr", "hi", "id", "pt",
                "ru", "ta", "th", "tr", "uk", "vi", "zh"]
    
    # idea: new processor; list of processors per-lang
    def _create_examples(self, objs, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, obj) in enumerate(objs):
            if obj['fold'] != set_type:
                continue
            if obj['lang'] != self.language and self.language != 'all':
                continue
            else:
                example_lang = obj['lang']
            guid = "%s-%s" % (set_type, i)
            # text_a = obj['title']
            sentences = sent_tokenize(obj['body'], lang=self.language)
            if len(sentences) >= 2:
                text_a = obj['title'] + sentences[0]
                text_b = sentences[1]
            elif len(sentences) == 1:
                text_a = obj['title'] + sentences[0]
                text_b = None
            else:
                text_a = obj['title']
                text_b = None
            label = obj['label']['about_tobacco']
            examples.append(
                ClassificationExample(guid=guid,
                                      text_a=text_a,
                                      text_b=text_b,
                                      label=label,
                                      language=example_lang))
        return examples


class MrpcProcessor(TSVProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        fp = f'{data_dir}/train.tsv'
        return self._create_examples(self._read_tsv(fp), "train")

    def get_dev_examples(self, data_dir):
        fp = f'{data_dir}/dev.tsv'
        return self._create_examples(self._read_tsv(fp), "dev")

    def get_test_examples(self, data_dir):
        fp = f'{data_dir}/test.tsv'
        return self._create_examples(self._read_tsv(fp), "test")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            if set_type == "test":
                label = "0"
            else:
                label = line[0]
            examples.append(
                ClassificationExample(guid=guid,
                                      text_a=text_a,
                                      text_b=text_b,
                                      label=label))
        return examples


class MnliProcessor(TSVProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        fp = f'{data_dir}/train.tsv'
        return self._create_examples(self._read_tsv(fp), "train")

    def get_dev_examples(self, data_dir):
        fp = f'{data_dir}/dev_matched.tsv'
        return self._create_examples(self._read_tsv(fp), "dev_matched")

    def get_test_examples(self, data_dir):
        fp = f'{data_dir}/test_matched.tsv'
        return self._create_examples(self._read_tsv(fp), "test")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            if set_type == "test":
                label = "contradiction"
            else:
                label = line[-1]
            examples.append(
                ClassificationExample(guid=guid,
                                      text_a=text_a,
                                      text_b=text_b,
                                      label=label))
        return examples


class ColaProcessor(TSVProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        fp = f'{data_dir}/train.tsv'
        return self._create_examples(self._read_tsv(fp), "train")

    def get_dev_examples(self, data_dir):
        fp = f'{data_dir}/dev.tsv'
        return self._create_examples(self._read_tsv(fp), "dev")

    def get_test_examples(self, data_dir):
        fp = f'{data_dir}/test.tsv'
        return self._create_examples(self._read_tsv(fp), "test")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = line[1]
                label = "0"
            else:
                text_a = line[3]
                label = line[1]
            examples.append(
                ClassificationExample(guid=guid,
                                      text_a=text_a,
                                      text_b=None,
                                      label=label))
        return examples


##################################################################
# LABELING
##################################################################


class LabelExample(object):
    """A single training/test example for simple sequence labeling."""

    def __init__(self, guid, text, label=None):
        """Constructs a LabelExample.

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class LabelProcessor(BaseProcessor):
    """Base class for data converters for sequence labeling data sets."""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, (text, label)) in enumerate(lines):
            guid = "%s-%d" % (set_type, i)
            examples.append(LabelExample(guid=guid, text=text, label=label))
        return examples


class NERProcessor(LabelProcessor):
    """Processor for the NER data set."""

    def __init__(self, lang):
        super().__init__()
        self.language = lang
        assert len(self.get_labels()) == len(set(self.get_labels()))
        self.id2label = self.get_labels()
        self.label2id = {l: i for i, l in enumerate(self.id2label)}

    def get_train_examples(self, data_dir):
        """See base class."""
        fp = f'{data_dir}/{self.language}/train.iob2.txt'
        return self._create_examples(self._read_file(fp), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        fp = f'{data_dir}/{self.language}/dev.iob2.txt'
        return self._create_examples(self._read_file(fp), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        fp = f'{data_dir}/{self.language}/test.iob2.txt'
        return self._create_examples(self._read_file(fp), "test")

    def get_labels(self):
        """See base class."""
        return [
            "B-LOC", "B-MISC", "B-ORG", "B-PER", "I-LOC", "I-MISC", "I-ORG",
            "I-PER", "O"
        ]

    @classmethod
    def _read_file(cls, input_file):
        """Reads an empty line seperated data (word \t label)."""
        with open(input_file, "r") as f:
            lines = []
            words, labels = [], []
            for line in f.readlines():
                line = line.strip()
                if not line:
                    assert len(words) == len(labels)
                    lines.append((words, labels))
                    words, labels = [], []
                else:
                    word, label = line.split('\t')
                    words.append(word)
                    labels.append(label)
            if len(words) == len(labels) and words:
                lines.append((words, labels))
            return lines


class WikiNERProcessor(NERProcessor):
    def get_train_examples(self, data_dir):
        fp = f'{data_dir}/{self.language}/train'
        return self._create_examples(self._read_file(fp), "train")

    def get_dev_examples(self, data_dir):
        fp = f'{data_dir}/{self.language}/dev'
        return self._create_examples(self._read_file(fp), "dev")

    def get_test_examples(self, data_dir):
        fp = f'{data_dir}/{self.language}/test'
        return self._create_examples(self._read_file(fp), "test")

    def get_labels(self):
        return ["B-LOC", "B-ORG", "B-PER", "I-LOC", "I-ORG", "I-PER", "O"]

    @classmethod
    def _read_file(cls, input_file):
        """Reads an empty line seperated data (word \t label)."""
        with open(input_file, "r") as f:
            lines = []
            words, labels = [], []
            for line in f.readlines():
                line = line.strip()
                if not line:
                    assert len(words) == len(labels)
                    lines.append((words, labels))
                    words, labels = [], []
                else:
                    word, label = line.split('\t')
                    word = word.split(':', 1)[1]
                    words.append(word)
                    labels.append(label)
            if len(words) == len(labels) and words:
                lines.append((words, labels))
            return lines


class POSProcessor(LabelProcessor):
    """Processor for the POS data set from UD."""

    def __init__(self, lang):
        super().__init__()
        self.language = lang
        assert len(self.get_labels()) == len(set(self.get_labels()))

    def get_train_examples(self, data_dir):
        """See base class."""
        fp = f'{data_dir}/{self.language}/{self.language}-ud-train.conllu'
        return self._create_examples(self._read_file(fp), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        fp = f'{data_dir}/{self.language}/{self.language}-ud-dev.conllu'
        return self._create_examples(self._read_file(fp), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        fp = f'{data_dir}/{self.language}/{self.language}-ud-test.conllu'
        return self._create_examples(self._read_file(fp), "test")

    def get_labels(self):
        """See base class."""
        return [
            "_", "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN",
            "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB",
            "X"
        ]

    @classmethod
    def _read_file(cls, input_file):
        """Reads an empty line seperated data (word \t label)."""
        with open(input_file, "r") as f:
            lines = []
            words, labels = [], []
            for line in f.readlines():
                tok = line.strip().split('\t')
                if len(tok) < 2 or line[0] == '#':
                    assert len(words) == len(labels)
                    if words:
                        lines.append((words, labels))
                        words, labels = [], []
                if tok[0].isdigit():
                    word, label = tok[1], tok[3]
                    words.append(word)
                    labels.append(label)
            if len(words) == len(labels) and words:
                lines.append((words, labels))
            return lines


class RawPOSProcessor(POSProcessor):
    '''
    UD 2.3 raw file
    '''

    def get_train_examples(self, data_dir):
        fp = f'{data_dir}/{self.language}/*-ud-train.conllu'
        fp = glob.glob(fp)
        if len(fp) == 1:
            return self._create_examples(self._read_file(fp[0]), "train")
        elif len(fp) == 0:
            return []
        else:
            raise ValueError('Duplicate train')

    def get_dev_examples(self, data_dir):
        fp = f'{data_dir}/{self.language}/*-ud-dev.conllu'
        fp = glob.glob(fp)
        if len(fp) == 1:
            return self._create_examples(self._read_file(fp[0]), "dev")
        elif len(fp) == 0:
            return []
        else:
            raise ValueError('Duplicate dev')

    def get_test_examples(self, data_dir):
        fp = f'{data_dir}/{self.language}/*-ud-test.conllu'
        fp = glob.glob(fp)
        if len(fp) == 1:
            return self._create_examples(self._read_file(fp[0]), "test")
        elif len(fp) == 0:
            return []
        else:
            raise ValueError('Duplicate test')

    def get_labels(self):
        return [
            "_", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
            "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB",
            "X"
        ]


##################################################################
# PARSING
##################################################################


class ParsingExample(object):
    """A single training/test example for simple sequence parsing."""

    def __init__(self, guid, text, pos, head=None, label=None):
        """Constructs a ParsingExample.

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            head: (Optional) int. Head of dependency relation
            label: (Optional) string. The label of the dependency relation. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.pos = pos
        self.head = head
        self.label = label


class ParsingProcessor(BaseProcessor):
    """Processor for the Parsing data set from UD."""

    def __init__(self, lang):
        super().__init__()
        self.language = lang
        assert len(self.get_labels()) == len(set(self.get_labels()))
        assert len(self.get_pos()) == len(set(self.get_pos()))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, (text, pos, head, label)) in enumerate(lines):
            guid = "%s-%d" % (set_type, i)
            examples.append(
                ParsingExample(guid=guid,
                               text=text,
                               pos=pos,
                               head=head,
                               label=label))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        fp = f"{data_dir}/{self.language}/{self.language}-ud-train.conllu"
        return self._create_examples(self._read_file(fp), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        fp = f"{data_dir}/{self.language}/{self.language}-ud-dev.conllu"
        return self._create_examples(self._read_file(fp), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        fp = f"{data_dir}/{self.language}/{self.language}-ud-test.conllu"
        return self._create_examples(self._read_file(fp), "test")

    def get_labels(self):
        return [
            '_', 'acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case',
            'cc', 'ccomp', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det',
            'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'goeswith',
            'iobj', 'list', 'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl',
            'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative',
            'xcomp', UNK
        ]

    def get_pos(self):
        return [
            "_", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
            "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
            "VERB", "X"
        ]

    @classmethod
    def _read_file(cls, input_file):
        """Reads an empty line seperated data (word \t label)."""
        with open(input_file, "r") as f:
            lines = []
            words, pos_tags, heads, labels = [], [], [], []
            for line in f.readlines():
                tok = line.strip().split('\t')
                if len(tok) < 2 or line[0] == '#':
                    assert len(words) == len(pos_tags) == len(heads) == len(
                        labels)
                    if words:
                        lines.append((words, pos_tags, heads, labels))
                        words, pos_tags, heads, labels = [], [], [], []
                if tok[0].isdigit():
                    word, pos, head, label = tok[1], tok[3], tok[6], tok[7]
                    words.append(word)
                    pos_tags.append(pos)
                    heads.append(int(head))
                    if pos in POS_TO_IGNORE:
                        labels.append('')
                    else:
                        labels.append(label.split(':')[0])
            if words:
                assert len(words) == len(pos_tags) == len(heads) == len(labels)
                lines.append((words, pos_tags, heads, labels))
            return lines


class RawParsingProcessor(ParsingProcessor):
    '''
    UD 2.3 raw file
    '''

    def get_labels(self):
        return [
            '_', 'acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case',
            'cc', 'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj', 'dep',
            'det', 'discourse', 'dislocated', 'expl', 'fixed', 'flat',
            'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nsubj', 'nummod',
            'obj', 'obl', 'orphan', 'parataxis', 'punct', 'reparandum', 'root',
            'vocative', 'xcomp', UNK
        ]

    def get_train_examples(self, data_dir):
        fp = f'{data_dir}/{self.language}/*-ud-train.conllu'
        fp = glob.glob(fp)
        if len(fp) == 1:
            return self._create_examples(self._read_file(fp[0]), "train")
        elif len(fp) == 0:
            return []
        else:
            raise ValueError('Duplicate train')

    def get_dev_examples(self, data_dir):
        fp = f'{data_dir}/{self.language}/*-ud-dev.conllu'
        fp = glob.glob(fp)
        if len(fp) == 1:
            return self._create_examples(self._read_file(fp[0]), "dev")
        elif len(fp) == 0:
            return []
        else:
            raise ValueError('Duplicate dev')

    def get_test_examples(self, data_dir):
        fp = f'{data_dir}/{self.language}/*-ud-test.conllu'
        fp = glob.glob(fp)
        if len(fp) == 1:
            return self._create_examples(self._read_file(fp[0]), "test")
        elif len(fp) == 0:
            return []
        else:
            raise ValueError('Duplicate test')
