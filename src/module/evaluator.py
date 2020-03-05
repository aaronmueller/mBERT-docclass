import re
from typing import Dict

import numpy as np
from sklearn import metrics
from allennlp.training.metrics import AttachmentScores


def convert_bio_to_spans(bio_sequence):
    spans = []  # (label, startindex, endindex)
    cur_start = None
    cur_label = None
    N = len(bio_sequence)
    for t in range(N + 1):
        if ((cur_start is not None)
                and (t == N or re.search("^[BO]", bio_sequence[t]))):
            assert cur_label is not None
            spans.append((cur_label, cur_start, t))
            cur_start = None
            cur_label = None
        if t == N: continue
        assert bio_sequence[t] and bio_sequence[t][0] in ("B", "I", "O")
        if bio_sequence[t].startswith("B"):
            cur_start = t
            cur_label = re.sub("^B-?", "", bio_sequence[t]).strip()
        if bio_sequence[t].startswith("I"):
            if cur_start is None:
                # warning(
                #     "BIO inconsistency: I without starting B. Rewriting to B.")
                newseq = bio_sequence[:]
                newseq[t] = "B" + newseq[t][1:]
                return convert_bio_to_spans(newseq)
            continuation_label = re.sub("^I-?", "", bio_sequence[t])
            if continuation_label != cur_label:
                newseq = bio_sequence[:]
                newseq[t] = "B" + newseq[t][1:]
                # warning(
                #     "BIO inconsistency: %s but current label is '%s'. Rewriting to %s"
                #     % (bio_sequence[t], cur_label, newseq[t]))
                return convert_bio_to_spans(newseq)

    # should have exited for last span ending at end by now
    assert cur_start is None
    return spans


class Evaluator(object):
    def add(self, gold, prediction):
        raise NotImplementedError

    def get_metric(self) -> Dict[str, float]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def unpack(self, gold, prediction):
        prediction = prediction.detach().cpu().numpy()
        gold = gold.cpu().numpy()
        return gold, prediction


class FullClassifierEvaluator(Evaluator):
    def __init__(self, average='binary', pos_label=1):
        self.average = average
        self.pos_label = pos_label
        self.gold = []
        self.prediction = []

    def add(self, gold, prediction):
        gold, prediction = self.unpack(gold, prediction)
        prediction = np.argmax(prediction, axis=1)
        self.gold.extend(gold.tolist())
        self.prediction.extend(prediction.tolist())

    def get_metric(self):
        acc = metrics.accuracy_score(self.gold, self.prediction)
        recall = metrics.recall_score(self.gold,
                                      self.prediction,
                                      average=self.average,
                                      pos_label=self.pos_label)
        precision = metrics.precision_score(self.gold,
                                            self.prediction,
                                            average=self.average,
                                            pos_label=self.pos_label)
        f1 = metrics.f1_score(self.gold,
                              self.prediction,
                              average=self.average,
                              pos_label=self.pos_label)
        return {'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1}

    def reset(self):
        self.gold = []
        self.prediction = []


class AccClassifierEvaluator(Evaluator):
    def __init__(self):
        self.gold = []
        self.prediction = []

    def add(self, gold, prediction):
        gold, prediction = self.unpack(gold, prediction)
        prediction = np.argmax(prediction, axis=1)
        self.gold.extend(gold.tolist())
        self.prediction.extend(prediction.tolist())

    def get_metric(self):
        acc = metrics.accuracy_score(self.gold, self.prediction)
        return {'acc': acc}

    def reset(self):
        self.gold = []
        self.prediction = []


class POSEvaluator(Evaluator):
    def __init__(self):
        self.num_correct = 0
        self.num_tokens = 0

    def add(self, gold, prediction):
        '''
        gold is label
        prediction is logits
        '''
        gold, prediction = self.unpack(gold, prediction)
        prediction = np.argmax(prediction, axis=-1)
        bs, seq_len = prediction.shape
        for ii in range(bs):
            for jj in range(seq_len):
                gold_label, pred_label = gold[ii, jj], prediction[ii, jj]
                if gold_label == -1:
                    continue
                if gold_label == pred_label:
                    self.num_correct += 1
                self.num_tokens += 1

    def get_metric(self):
        return {'acc': self.num_correct / self.num_tokens}

    def reset(self):
        self.num_correct = 0
        self.num_tokens = 0


class NEREvaluator(Evaluator):
    def __init__(self, processor):
        self.processor = processor

        self.tp, self.fp, self.fn = 0, 0, 0

    def add(self, gold, prediction):
        '''
        gold is label
        prediction is logits
        '''
        gold, prediction = self.unpack(gold, prediction)
        prediction = np.argmax(prediction, axis=-1)
        bs, seq_len = prediction.shape
        for ii in range(bs):
            goldseq, predseq = [], []
            for jj in range(seq_len):
                gold_label, pred_label = gold[ii, jj], prediction[ii, jj]
                if gold_label == -1:
                    continue
                goldseq.append(self.processor.id2label[gold_label])
                predseq.append(self.processor.id2label[pred_label])

            goldspans = convert_bio_to_spans(goldseq)
            predspans = convert_bio_to_spans(predseq)

            goldspans_set = set(goldspans)
            predspans_set = set(predspans)

            # tp: number of spans that gold and pred have
            # fp: number of spans that pred had that gold didn't (incorrect predictions)
            # fn: number of spans that gold had that pred didn't (didn't recall)
            self.tp += len(goldspans_set & predspans_set)
            self.fp += len(predspans_set - goldspans_set)
            self.fn += len(goldspans_set - predspans_set)

    def get_metric(self):
        try:
            prec = self.tp / (self.tp + self.fp)
            rec = self.tp / (self.tp + self.fn)
            f1 = 2 * prec * rec / (prec + rec)
        except:
            f1 = 0
        return {'f1': f1}

    def reset(self):
        self.tp, self.fp, self.fn = 0, 0, 0


class ParsingEvaluator(Evaluator):
    def __init__(self):
        self._attachment_scores = AttachmentScores()

    def add(self, gold_idx, gold_label, prediction_idx, prediction_label,
            mask):
        self._attachment_scores(prediction_idx, prediction_label, gold_idx,
                                gold_label, mask)

    def get_metric(self):
        return self._attachment_scores.get_metric(reset=False)

    def reset(self):
        self._attachment_scores.reset()