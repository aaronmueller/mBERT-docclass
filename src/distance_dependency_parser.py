import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules import FeedForward, InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import \
    BilinearMatrixAttention
from allennlp.nn import Activation
from allennlp.nn.chu_liu_edmonds import decode_mst
from allennlp.nn.util import (get_device_of,
                              get_lengths_from_binary_sequence_mask,
                              get_range_vector, masked_log_softmax)
from allennlp.training.metrics.metric import Metric
from overrides import overrides

from bert.modeling import BertModel, PreTrainedBertModel, freeze


@Metric.register("undirected_attachment_scores")
class UndirectedAttachmentScores(Metric):
    """
    Computes labeled and unlabeled attachment scores for a
    dependency parse, as well as sentence level exact match
    for both labeled and unlabeled trees. Note that the input
    to this metric is the sampled predictions, not the distribution
    itself.

    Parameters
    ----------
    ignore_classes : ``List[int]``, optional (default = None)
        A list of label ids to ignore when computing metrics.
    """

    def __init__(self, ignore_classes: List[int] = None) -> None:
        self._unlabeled_correct = 0.
        self._total_words = 0.

        self._ignore_classes: List[int] = ignore_classes or []

    def __call__(
            self,  # type: ignore
            predicted_indices: torch.Tensor,
            gold_indices: torch.Tensor,
            mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predicted_indices : ``torch.Tensor``, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        gold_indices : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_indices``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predicted_indices``.
        """
        unwrapped = self.unwrap_to_tensors(predicted_indices, gold_indices,
                                           mask)
        predicted_indices, gold_indices, mask = unwrapped

        mask = mask.long()
        predicted_indices = predicted_indices.long()
        gold_indices = gold_indices.long()

        # # Multiply by a mask donoting locations of
        # # gold labels which we should ignore.
        # for label in self._ignore_classes:
        #     label_mask = gold_labels.eq(label)
        #     mask = mask * (1 - label_mask).long()

        correct_indices = predicted_indices.eq(gold_indices).long() * mask

        self._unlabeled_correct += correct_indices.sum()
        self._total_words += correct_indices.numel() - (1 - mask).sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated metrics as a dictionary.
        """
        unlabeled_attachment_score = 0.0
        if self._total_words > 0.0:
            unlabeled_attachment_score = float(
                self._unlabeled_correct) / float(self._total_words)
        if reset:
            self.reset()
        return {
            "UUAS": unlabeled_attachment_score,
        }

    @overrides
    def reset(self):
        self._unlabeled_correct = 0.
        self._total_words = 0.


class DistanceAttention(torch.nn.Module):
    """
    ``DistanceAttention`` takes one matrices as input and returns a matrix of attentions.

    We compute the similarity between each row in each matrix and return unnormalized similarity
    scores.  Because these scores are unnormalized, we don't take a mask as input; it's up to the
    caller to deal with masking properly when this output is used.

    Input:
        - matrix_1: ``(batch_size, num_rows_1, embedding_dim_1)``

    Output:
        - ``(batch_size, num_rows_1, num_rows_1)``
    """

    def forward(
            self,  # pylint: disable=arguments-differ
            matrix_1: torch.Tensor) -> torch.Tensor:
        batchlen, seqlen, rank = matrix_1.size()
        matrix_1 = matrix_1.unsqueeze(2)
        matrix_1 = matrix_1.expand(-1, -1, seqlen, -1)
        transposed = matrix_1.transpose(1, 2)
        diffs = matrix_1 - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances


class DistanceDependencyParser(PreTrainedBertModel):
    """
    This dependency parser follows the model of
    ` Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)
    <https://arxiv.org/abs/1611.01734>`_ .

    Word representations are generated using a bidirectional LSTM,
    followed by separate biaffine classifiers for pairs of words,
    predicting whether a directed arc exists between the two words
    and the dependency label the arc should have. Decoding can either
    be done greedily, or the optimal Minimum Spanning Tree can be
    decoded using Edmond's algorithm by viewing the dependency tree as
    a MST on a fully connected graph, where nodes are words and edges
    are scored dependency arcs.

    Parameters
    ----------
    arc_representation_dim : ``int``, required.
        The dimension of the MLPs used for head arc prediction.
    arc_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    use_mst_decoding_for_validation : ``bool``, optional (default = True).
        Whether to use Edmond's algorithm to find the optimal minimum spanning tree during validation.
        If false, decoding is greedy.
    FIXME: Improving https://nlp.stanford.edu/pubs/hewitt2019structural.pdf with direct loss function
    """

    def __init__(self,
                 config,
                 num_labels: int,
                 num_pos: int,
                 use_pos: bool,
                 arc_representation_dim: int,
                 arc_feedforward: FeedForward = None,
                 use_mst_decoding_for_validation: bool = True,
                 dropout: float = 0.) -> None:
        super(DistanceDependencyParser, self).__init__(config)
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

        encoder_dim = config.hidden_size

        self.arc_feedforward = arc_feedforward or \
                                    FeedForward(encoder_dim, 1,
                                                arc_representation_dim,
                                                Activation.by_name("linear")())

        self.arc_attention = DistanceAttention()

        self._dropout = InputVariationalDropout(dropout)

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation

        self._attachment_scores = UndirectedAttachmentScores()
        # initializer(self)

    def encode(self, input_ids, token_type_ids, attention_mask):
        sequence_output, _ = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False)
        return sequence_output

    def forward(
            self,  # type: ignore
            input_ids: torch.LongTensor,
            pos_ids: torch.LongTensor,
            segment_ids: torch.LongTensor,
            input_mask: torch.LongTensor,
            nonword_mask: torch.LongTensor,
            head_tags: torch.LongTensor = None,
            head_indices: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        input_ids: torch.LongTensor, required. Has shape ``(batch_size, sequence_length)``.
        segment_ids: torch.LongTensor, required. Has shape ``(batch_size, sequence_length)``.
        input_mask: torch.LongTensor, required. Has shape ``(batch_size, sequence_length)``.
        nonword_mask: torch.LongTensor, required. Has shape ``(batch_size, sequence_length)``.
        head_indices : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape ``(batch_size, sequence_length)``.

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        arc_loss : ``torch.FloatTensor``
            The loss contribution from the unlabeled arcs.
        loss : ``torch.FloatTensor``, optional
            The loss contribution from predicting the dependency
            tags for the gold arcs.
        heads : ``torch.FloatTensor``
            The predicted head indices for each word. A tensor
            of shape (batch_size, sequence_length).
        head_types : ``torch.FloatTensor``
            The predicted head types for each arc. A tensor
            of shape (batch_size, sequence_length).
        mask : ``torch.LongTensor``
            A mask denoting the padded elements in the batch.
        """
        encoded_text = self.encode(input_ids, segment_ids, input_mask)
        # batch_size, _, encoding_dim = encoded_text.size()

        float_mask = nonword_mask.float()
        encoded_text = self._dropout(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        arc_representation = self._dropout(self.arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(arc_representation)

        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(
            2) + minus_mask.unsqueeze(1)

        if self.training or not self.use_mst_decoding_for_validation:
            predicted_heads = self._greedy_decode(attended_arcs, nonword_mask)
        else:
            lengths = input_mask.data.sum(dim=1).long().cpu().numpy()
            predicted_heads = self._mst_decode(attended_arcs, nonword_mask,
                                               lengths)
        if head_indices is not None:

            loss = self._construct_loss(
                attended_arcs=attended_arcs,
                head_indices=head_indices,
                mask=nonword_mask)

            # We calculate attatchment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            self._attachment_scores(predicted_heads[:, 1:],
                                    head_indices[:, 1:], nonword_mask[:, 1:])
        else:
            loss = self._construct_loss(
                attended_arcs=attended_arcs,
                head_indices=predicted_heads.long(),
                mask=nonword_mask)

        # output_dict = {
        #     "heads": predicted_heads,
        #     "head_tags": predicted_head_tags,
        #     "arc_loss": arc_nll,
        #     "tag_loss": tag_nll,
        #     "loss": loss,
        #     "mask": mask,
        #     "words": [meta["words"] for meta in metadata],
        #     "pos": [meta["pos"] for meta in metadata]
        # }
        return loss

    def decode(self, output_dict: Dict[str, torch.Tensor]
               ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
        heads = output_dict.pop("heads").cpu().detach().numpy()
        mask = output_dict.pop("mask")
        lengths = get_lengths_from_binary_sequence_mask(mask)
        head_indices = []
        for instance_heads, length in zip(heads, lengths):
            instance_heads = list(instance_heads[1:length])
            head_indices.append(instance_heads)

        output_dict["predicted_heads"] = head_indices
        return output_dict

    def _construct_loss(
            self, attended_arcs: torch.Tensor, head_indices: torch.Tensor,
            mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc tag loss.
        """
        float_mask = mask.float()
        batch_size, sequence_length, _ = attended_arcs.size()
        # shape (batch_size, 1)
        range_vector = get_range_vector(
            batch_size, get_device_of(attended_arcs)).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits = masked_log_softmax(
            attended_arcs,
            mask) * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)

        # index matrix with shape (batch, sequence_length)
        timestep_index = get_range_vector(sequence_length,
                                          get_device_of(attended_arcs))
        child_index = timestep_index.view(1, sequence_length).expand(
            batch_size, sequence_length).long()
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index,
                                         head_indices]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        return arc_nll

    def _greedy_decode(self, attended_arcs: torch.Tensor, mask: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        Parameters
        ----------
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = attended_arcs + torch.diag(
            attended_arcs.new(mask.size(1)).fill_(-numpy.inf))
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = (1 - mask).byte().unsqueeze(2)
            attended_arcs.masked_fill_(minus_mask, -numpy.inf)

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = attended_arcs.max(dim=2)
        return heads

    def _mst_decode(
            self, attended_arcs: torch.Tensor, mask: torch.Tensor,
            lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.

        Parameters
        ----------
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """
        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = (1 - mask.float()) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(
            2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = F.log_softmax(
            attended_arcs, dim=2).transpose(1, 2)

        # Shape (batch_size, 1, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy = torch.exp(normalized_arc_logits.unsqueeze(1))
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(batch_energy: torch.Tensor, lengths: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, _ = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(
                scores.numpy(), length, has_labels=False)

            # We don't care what the head or tag is for the root token, but by default it's
            # not necesarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            heads.append(instance_heads)
        return torch.from_numpy(numpy.stack(heads))

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._attachment_scores.get_metric(reset)

    def reset_metrics(self):
        return self._attachment_scores.reset()


class DistanceDependencyParserFeatureAtX(DistanceDependencyParser):
    def __init__(self, config, num_labels, num_pos, use_pos,
                 arc_representation_dim, feature_layer_idx):
        super(DistanceDependencyParserFeatureAtX, self).__init__(
            config, num_labels, num_pos, use_pos, arc_representation_dim)
        self.feature_layer_idx = feature_layer_idx
        freeze(self.bert.embeddings)
        for layer in self.bert.encoder.layer:
            freeze(layer)

    def encode(self, input_ids, token_type_ids, attention_mask):
        encoder_output, _ = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            first_nb_layers=self.feature_layer_idx)
        return encoder_output
