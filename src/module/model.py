import copy
from dataclasses import dataclass
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
from torch import Tensor

from .component import BaseModel, BaseModelConfig, BertInput, PoolDecoderConfig, MeanPoolDecoderConfig


@dataclass
class SeqClassConfig(BaseModelConfig):
    num_labels: int = -1

    def build(self):
        return SeqClass(self)


class SeqClass(BaseModel):
    def __init__(self, config: SeqClassConfig):
        assert isinstance(config, SeqClassConfig)
        assert (isinstance(config.decoder_config, PoolDecoderConfig)
                or isinstance(config.decoder_config, MeanPoolDecoderConfig))
        super().__init__(config)
        self.config = config
        self.classifier = nn.Linear(config.decoder_config.output_dim,
                                    config.num_labels)

    def forward(self,
                token: Tensor,
                token_type: Tensor,
                position: Tensor,
                mask: Tensor,
                label: Tensor,
                ex_lang: Tensor,
                lang_key: Optional[str] = None):
        inputs = BertInput(token=token,
                           token_type=token_type,
                           position=position,
                           mask=mask,
                           lang_key=lang_key)
        ctx = self.embed(inputs)
        #ctx = ctx.cpu()
        logits = F.log_softmax(self.classifier(ctx), dim=-1)
        loss = F.nll_loss(logits, label)
        self.evaluator.add(label, logits)
        return loss


@dataclass
class SeqLabelConfig(BaseModelConfig):
    num_labels: int = -1
    label_pad_idx: int = -1

    def build(self):
        return SeqLabel(self)


class SeqLabel(BaseModel):
    def __init__(self, config: SeqLabelConfig):
        assert isinstance(config, SeqLabelConfig)
        super().__init__(config)
        self.config = config
        self.classifier = nn.Linear(config.decoder_config.output_dim,
                                    config.num_labels)

    def forward(self,
                token: Tensor,
                token_type: Tensor,
                position: Tensor,
                mask: Tensor,
                label: Tensor,
                ex_lang: Tensor,
                lang_key: Optional[str] = None):
        inputs = BertInput(token=token,
                           token_type=token_type,
                           position=position,
                           mask=mask,
                           lang_key=lang_key)
        ctx = self.embed(inputs)
        #ctx = ctx.cpu()
        logits = F.log_softmax(self.classifier(ctx), dim=-1)
        loss = F.nll_loss(logits.view(-1, self.config.num_labels),
                          label.view(-1),
                          ignore_index=self.config.label_pad_idx)
        self.evaluator.add(label, logits)
        return loss

# similar to above, but with separate top-level classification layer
@dataclass
class SeqSepTopClassConfig(BaseModelConfig):
    num_labels: int = -1

    def build(self):
        return SeqSepTopClass(self)


class SeqSepTopClass(BaseModel):
    def __init__(self, config: SeqSepTopClassConfig):
        assert isinstance(config, SeqSepTopClassConfig)
        assert (isinstance(config.decoder_config, PoolDecoderConfig)
                or isinstance(config.decoder_config, MeanPoolDecoderConfig))
        super().__init__(config)
        self.config = config
        self.classifier = {}

    def forward(self,
                token: Tensor,
                token_type: Tensor,
                position: Tensor,
                mask: Tensor,
                label: Tensor,
                ex_lang: Tensor,
                lang_key: Optional[str] = None):
        inputs = BertInput(token=token,
                           token_type=token_type,
                           position=position,
                           mask=mask,
                           lang_key=lang_key)
        ctx = self.embed(inputs)
        ctx = ctx.cpu()
        if ex_lang.item() not in self.classifier.keys():
            self.classifier[ex_lang.item()] = nn.Linear(self.config.decoder_config.output_dim,
                                                  self.config.num_labels)
        logits = F.log_softmax(self.classifier[ex_lang.item()](ctx), dim=-1)
        loss = F.nll_loss(logits, label.cpu())
        self.evaluator.add(label.cpu(), logits)
        return loss


@dataclass
class SeqSepTopLabelConfig(BaseModelConfig):
    num_labels: int = -1
    label_pad_idx: int = -1

    def build(self):
        return SeqSepTopLabel(self)


class SeqSepTopLabel(BaseModel):
    def __init__(self, config: SeqSepTopLabelConfig):
        assert isinstance(config, SeqSepTopLabelConfig)
        super().__init__(config)
        self.config = config
        self.classifier = {}

    def forward(self,
                token: Tensor,
                token_type: Tensor,
                position: Tensor,
                mask: Tensor,
                label: Tensor,
                ex_lang: Tensor,
                lang_key: Optional[str] = None):
        inputs = BertInput(token=token,
                           token_type=token_type,
                           position=position,
                           mask=mask,
                           lang_key=lang_key)
        ctx = self.embed(inputs)
        ctx = ctx.cpu()
        if ex_lang.item() not in self.classifier.keys():
            self.classifier[ex_lang.item()] = nn.Linear(self.config.decoder_config.output_dim,
                                                  self.config.num_labels)
        logits = F.log_softmax(self.classifier[ex_lang.item()](ctx), dim=-1)
        loss = F.nll_loss(logits.view(-1, self.config.num_labels),
                          label.view(-1).cpu(),
                          ignore_index=self.config.label_pad_idx)
        self.evaluator.add(label.cpu(), logits)
        return loss


@dataclass
class BaselineSeqClassConfig(BaseModelConfig):
    num_labels: int = -1

    def build(self):
        return BaselineSeqClass(self)


class BaselineSeqClass(BaseModel):
    def __init__(self, config: BaselineSeqClassConfig):
        assert isinstance(config, BaselineSeqClassConfig)
        assert (isinstance(config.decoder_config, PoolDecoderConfig)
                or isinstance(config.decoder_config, MeanPoolDecoderConfig))
        super().__init__(config)
        self.config = config
        self.embedding = nn.EmbeddingBag(119547, 768, mode='mean')
        self.classifier = nn.Linear(config.decoder_config.output_dim,
                                    config.num_labels)
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self,
                token: Tensor,
                token_type: Tensor,
                position: Tensor,
                mask: Tensor,
                label: Tensor,
                ex_lang: Tensor,
                lang_key: Optional[str] = None):
        ctx = self.embedding(token)
        #ctx = ctx.cpu()
        logits = F.log_softmax(self.classifier(ctx), dim=-1)
        loss = F.nll_loss(logits, label)
        self.evaluator.add(label, logits)
        return loss


@dataclass
class BaselineSeqLabelConfig(BaseModelConfig):
    num_labels: int = -1
    label_pad_idx: int = -1

    def build(self):
        return BaselineSeqLabel(self)


class BaselineSeqLabel(BaseModel):
    def __init__(self, config: BaselineSeqLabelConfig):
        assert isinstance(config, BaselineSeqLabelConfig)
        super().__init__(config)
        self.config = config
        self.embedding = nn.EmbeddingBag(119547, 768, mode='mean')
        self.classifier = nn.Linear(config.decoder_config.output_dim,
                                    config.num_labels)
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self,
                token: Tensor,
                token_type: Tensor,
                position: Tensor,
                mask: Tensor,
                label: Tensor,
                ex_lang: Tensor,
                lang_key: Optional[str] = None):
        ctx = self.embedding(inputs)
        #ctx = ctx.cpu()
        logits = F.log_softmax(self.classifier(ctx), dim=-1)
        loss = F.nll_loss(logits.view(-1, self.config.num_labels),
                          label.view(-1),
                          ignore_index=self.config.label_pad_idx)
        self.evaluator.add(label, logits)
        return loss


@dataclass
class ParsingConfig(BaseModelConfig):
    num_labels: int = -1
    num_pos: int = -1
    use_pos: bool = False
    pos_dim: int = 100
    tag_dim: int = 128
    arc_dim: int = 512
    use_mst_decoding_for_validation: bool = True
    dropout: float = 0.33

    def build(self):
        return BiaffineDependencyParser(self)


class BiaffineDependencyParser(BaseModel):
    """
    This dependency parser follows the model of
    ` Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)
    <https://arxiv.org/abs/1611.01734>`_ . (Based on AllenNLP)
    """
    def __init__(self, config: ParsingConfig):
        assert isinstance(config, ParsingConfig)
        super().__init__(config)
        self.config = config
        encoder_dim = config.decoder_config.output_dim

        if self.config.use_pos:
            self.pos_embedding = nn.Embedding(config.num_pos,
                                              config.pos_dim,
                                              padding_idx=0)
            encoder_dim += config.pos_dim

        self.head_arc_feedforward = FeedForward(encoder_dim, 1, config.arc_dim,
                                                Activation.by_name("elu")())
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(config.arc_dim,
                                                     config.arc_dim,
                                                     use_input_biases=True)

        self.head_tag_feedforward = FeedForward(encoder_dim, 1, config.tag_dim,
                                                Activation.by_name("elu")())
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = torch.nn.modules.Bilinear(config.tag_dim,
                                                      config.tag_dim,
                                                      config.num_labels)
        self.dropout = InputVariationalDropout(config.dropout)
        self.use_mst_decoding_for_validation = config.use_mst_decoding_for_validation

    def forward(self,
                input_ids: Tensor,
                pos_ids: Tensor,
                segment_ids: Tensor,
                position: Tensor,
                input_mask: Tensor,
                nonword_mask: Tensor,
                head_tags: Tensor,
                head_indices: Tensor,
                lang_key: Optional[str] = None):
        """
        Parameters
        ----------
        input_ids: torch.LongTensor, required. Has shape ``(batch_size, sequence_length)``.
        input_mask: torch.LongTensor, required. Has shape ``(batch_size, sequence_length)``.
        nonword_mask: torch.LongTensor, required. Has shape ``(batch_size, sequence_length)``.
        segment_ids: torch.LongTensor, required. Has shape ``(batch_size, sequence_length)``.
        head_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels for the arcs
            in the dependency parse. Has shape ``(batch_size, sequence_length)``.
        head_indices : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape ``(batch_size, sequence_length)``.
        """
        inputs = BertInput(token=input_ids,
                           token_type=segment_ids,
                           position=position,
                           mask=input_mask,
                           lang_key=lang_key)
        encoded_text = self.embed(inputs)

        if self.config.use_pos:
            encoded_pos = self.decoder.dropout(self.pos_embedding(pos_ids))
            encoded_text = torch.cat((encoded_text, encoded_pos), dim=-1)

        batch_size, _, encoding_dim = encoded_text.size()

        float_mask = nonword_mask.float()

        # shape (batch_size, sequence_length, arc_dim)
        head_arc_representation = self.dropout(
            self.head_arc_feedforward(encoded_text))
        child_arc_representation = self.dropout(
            self.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, tag_dim)
        head_tag_representation = self.dropout(
            self.head_tag_feedforward(encoded_text))
        child_tag_representation = self.dropout(
            self.child_tag_feedforward(encoded_text))
        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(head_arc_representation,
                                           child_arc_representation)

        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(
            2) + minus_mask.unsqueeze(1)

        if self.training or not self.use_mst_decoding_for_validation:
            predicted_heads, predicted_head_tags = self._greedy_decode(
                head_tag_representation, child_tag_representation,
                attended_arcs, nonword_mask)
        else:
            lengths = input_mask.data.sum(dim=1).long().cpu().numpy()
            predicted_heads, predicted_head_tags = self._mst_decode(
                head_tag_representation, child_tag_representation,
                attended_arcs, nonword_mask, lengths)

        arc_nll, tag_nll = self._construct_loss(
            head_tag_representation=head_tag_representation,
            child_tag_representation=child_tag_representation,
            attended_arcs=attended_arcs,
            head_indices=head_indices,
            head_tags=head_tags,
            mask=nonword_mask)
        loss = arc_nll + tag_nll

        # We calculate attatchment scores for the whole sentence
        # but excluding the symbolic ROOT token at the start,
        # which is why we start from the second element in the sequence.
        self.evaluator.add(head_indices[:, 1:], head_tags[:, 1:],
                           predicted_heads[:, 1:], predicted_head_tags[:, 1:],
                           nonword_mask[:, 1:])
        return loss

    def _construct_loss(self, head_tag_representation: torch.Tensor,
                        child_tag_representation: torch.Tensor,
                        attended_arcs: torch.Tensor,
                        head_indices: torch.Tensor, head_tags: torch.Tensor,
                        mask: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
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

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag_representation,
                                              child_tag_representation,
                                              head_indices)
        normalised_head_tag_logits = masked_log_softmax(
            head_tag_logits, mask.unsqueeze(-1)) * float_mask.unsqueeze(-1)
        # index matrix with shape (batch, sequence_length)
        timestep_index = get_range_vector(sequence_length,
                                          get_device_of(attended_arcs))
        child_index = timestep_index.view(1, sequence_length).expand(
            batch_size, sequence_length).long()
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index,
                                         head_indices]
        tag_loss = normalised_head_tag_logits[range_vector, child_index,
                                              head_tags]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        tag_nll = -tag_loss.sum() / valid_positions.float()
        return arc_nll, tag_nll

    def _greedy_decode(self, head_tag_representation: torch.Tensor,
                       child_tag_representation: torch.Tensor,
                       attended_arcs: torch.Tensor, mask: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
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

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag_representation,
                                              child_tag_representation, heads)
        _, head_tags = head_tag_logits.max(dim=2)
        return heads, head_tags

    def _mst_decode(self, head_tag_representation: torch.Tensor,
                    child_tag_representation: torch.Tensor,
                    attended_arcs: torch.Tensor, mask: torch.Tensor,
                    lengths: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
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
        batch_size, sequence_length, tag_dim = head_tag_representation.size()

        # lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [
            batch_size, sequence_length, sequence_length, tag_dim
        ]
        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(
            *expanded_shape).contiguous()
        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(
            *expanded_shape).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.tag_bilinear(head_tag_representation,
                                                 child_tag_representation)

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels,sequence_length, sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits,
                                                        dim=3).permute(
                                                            0, 3, 1, 2)

        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = (1 - mask.float()) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(
            2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = F.log_softmax(attended_arcs,
                                              dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy = torch.exp(
            normalized_arc_logits.unsqueeze(1) +
            normalized_pairwise_head_logits)
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(batch_energy: torch.Tensor, lengths: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(),
                                           length,
                                           has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necesarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)
        return torch.from_numpy(numpy.stack(heads)), torch.from_numpy(
            numpy.stack(head_tags))

    def _get_head_tags(self, head_tag_representation: torch.Tensor,
                       child_tag_representation: torch.Tensor,
                       head_indices: torch.Tensor) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every word.

        Returns
        -------
        head_tag_logits : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_tag_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(
            batch_size, get_device_of(head_tag_representation)).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_dim)
        selected_head_tag_representations = head_tag_representation[
            range_vector, head_indices]
        selected_head_tag_representations = selected_head_tag_representations.contiguous(
        )
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(selected_head_tag_representations,
                                            child_tag_representation)
        return head_tag_logits
