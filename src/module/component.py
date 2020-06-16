from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from allennlp.modules import InputVariationalDropout
# from pytorch_transformers import BertModel, XLMModel
from pytorch_transformers import BertModel
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

from .modeling_xlm import XLMModel


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


@dataclass
class BaseEncoderConfig:
    # freeze bottom X layers, including X-th layer
    freeze_layer: int = -1

    def build(self):
        return BaseEncoder(self)


@dataclass
class BaseInput:
    pass


@dataclass
class EncoderOutput:
    encoder_layers: List[Tensor]
    pool_output: Tensor
    attention_mask: Tensor

    def __post_init__(self):
        assert isinstance(self.encoder_layers, tuple)
        for h in self.encoder_layers:
            assert isinstance(h, Tensor)
        assert isinstance(self.pool_output, Tensor)
        assert isinstance(self.attention_mask, Tensor)
        assert self.pool_output.shape[0] == self.encoder_layers[0].shape[0]
        assert self.pool_output.shape[0] == self.attention_mask.shape[0]
        for h in self.encoder_layers:
            assert self.encoder_layers[0].shape == h.shape


class BaseEncoder(nn.Module):
    def __init__(self, config: BaseEncoderConfig):
        assert isinstance(config, BaseEncoderConfig)
        super().__init__()
        self.config = config

    def freeze_layers(self):
        if self.config.freeze_layer == -1:
            return
        elif self.config.freeze_layer >= 0:
            for i in range(self.config.freeze_layer + 1):
                if i == 0:
                    self.freeze_embeddings()
                else:
                    self.freeze_layer(i)

    def freeze_embeddings(self):
        raise NotImplementedError

    def freeze_layer(self, layer):
        raise NotImplementedError

    def encode(self, inputs: BaseInput) -> EncoderOutput:
        '''
        return encoder all hidden state [batch_size, sequence_length, hidden_size] and
        pooled hidden state [batch_size, hidden_size]
        '''
        raise NotImplementedError

    def get_hidden_dim(self) -> int:
        raise NotImplementedError

    def get_num_layers(self) -> int:
        raise NotImplementedError

'''
@dataclass
class BaselineEncoderConfig(BaseEncoderConfig):
    mode: str = 'logistic-regression'

    def build(self):
        return BaselineEncoder(self)

class BaselineInput(BaseInput):
    token: Tensor
    token_type: Tensor
    position: Tensor
    mask: Tensor
    lang_key: Optional[str] = None

class BaselineEncoder(BaseEncoder):
    def __init__(self, config: BaselineEncoderConfig = BaselineEncoderConfig()):
        VOCAB_SIZE = 30000
        EMBED_DIM = 512
        assert isinstance(config, BaselineEncoderConfig)
        super().__init__(config)
        self.config = config
        self.embedding = nn.EmbeddingBag(VOCAB_SIZE, EMBED_DIM, sparse=True)
'''

@dataclass
class BertEncoderConfig(BaseEncoderConfig):
    mode: str = 'bert-base-multilingual-cased'

    def build(self):
        return BertEncoder(self)


@dataclass
class BertInput(BaseInput):
    token: Tensor
    token_type: Tensor
    position: Tensor
    mask: Tensor
    lang_key: Optional[str] = None


class BertEncoder(BaseEncoder):
    def __init__(self, config: BertEncoderConfig = BertEncoderConfig()):
        assert isinstance(config, BertEncoderConfig)
        super().__init__(config)
        self.config = config
        self.model = BertModel.from_pretrained(self.config.mode,
                                               output_hidden_states=True)
        self.freeze_layers()

    def freeze_embeddings(self):
        freeze(self.model.embeddings)

    def freeze_layer(self, layer):
        freeze(self.model.encoder.layer[layer - 1])

    def encode(self, inputs: BertInput) -> EncoderOutput:
        _, pooled_output, encoded_layers = self.model.forward(
            inputs.token,
            token_type_ids=inputs.token_type,
            attention_mask=inputs.mask,
            position_ids=inputs.position)
        return EncoderOutput(encoded_layers, pooled_output, inputs.mask)

    def get_hidden_dim(self) -> int:
        return self.model.config.hidden_size

    def get_num_layers(self) -> int:
        return self.model.config.num_hidden_layers + 1


@dataclass
class XLMEncoderConfig(BaseEncoderConfig):
    mode: str = 'xlm-mlm-tlm-xnli15-1024'

    def build(self):
        return XLMEncoder(self)


class XLMEncoder(BaseEncoder):
    def __init__(self, config: XLMEncoderConfig = XLMEncoderConfig()):
        assert isinstance(config, XLMEncoderConfig)
        super().__init__(config)
        self.config = config
        self.model = XLMModel.from_pretrained(self.config.mode,
                                              output_hidden_states=True)
        self.freeze_layers()

    def freeze_embeddings(self):
        freeze(self.model.position_embeddings)
        if self.model.n_langs > 1 and self.model.use_lang_emb:
            freeze(self.model.lang_embeddings)
        freeze(self.model.embeddings)

    def freeze_layer(self, layer):
        freeze(self.model.attentions[layer - 1])
        freeze(self.model.layer_norm1[layer - 1])
        freeze(self.model.ffns[layer - 1])
        freeze(self.model.layer_norm2[layer - 1])

    def encode(self, inputs: BertInput) -> EncoderOutput:
        last_layer, encoded_layers = self.model.forward(
            inputs.token,
            position_ids=inputs.position,
            langs=inputs.token_type,
            attention_mask=inputs.mask,
            lang_key=inputs.lang_key)
        pooled_output = last_layer[:, 0]
        return EncoderOutput(encoded_layers, pooled_output, inputs.mask)

    def get_hidden_dim(self) -> int:
        return self.model.dim

    def get_num_layers(self) -> int:
        return self.model.n_layers + 1


@dataclass
class BaseDecoderConfig:
    # input dim is filled in by encoder
    input_dropout: float = 0.2
    input_dim: int = -1
    output_dim: int = -1
    feature_layer: int = -1
    num_encoder_layers: int = -1
    weighted_feature: bool = False

    # def __post_init__(self):
    #     assert self.num_encoder_layers > 0
    #     assert -1 <= self.feature_layer < self.num_encoder_layers

    def build(self):
        return BaseDecoder(self)


class BaseDecoder(nn.Module):
    def __init__(self, config: BaseDecoderConfig):
        assert isinstance(config, BaseDecoderConfig)
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.zeros(config.num_encoder_layers))
        self.mapping: nn.Module = nn.Identity()
        self.dropout = InputVariationalDropout(config.input_dropout)

    def preprocess(self, hidden_state: EncoderOutput) -> Tensor:
        if self.config.weighted_feature:
            encoder_layers = torch.stack(hidden_state.encoder_layers)
            encoder_output = encoder_layers * F.softmax(
                self.weight, dim=0).view(-1, 1, 1, 1)
            encoder_output = encoder_output.sum(dim=0)
        else:
            encoder_output = hidden_state.encoder_layers[self.config.feature_layer]
        return self.dropout(encoder_output)

    def decode(self, hidden_state: EncoderOutput) -> Tensor:
        return self.mapping(self.preprocess(hidden_state))


@dataclass
class PoolDecoderConfig(BaseDecoderConfig):
    def build(self):
        return PoolDecoder(self)


class PoolDecoder(BaseDecoder):
    def __init__(self, config: PoolDecoderConfig):
        assert isinstance(config, PoolDecoderConfig)
        super().__init__(config)
        self.dropout = nn.Dropout(config.input_dropout)

    def preprocess(self, hidden_state: EncoderOutput) -> Tensor:
        return self.dropout(hidden_state.pool_output)


@dataclass
class MeanPoolDecoderConfig(BaseDecoderConfig):
    def build(self):
        return MeanPoolDecoder(self)


class MeanPoolDecoder(BaseDecoder):
    def __init__(self, config: MeanPoolDecoderConfig):
        assert isinstance(config, MeanPoolDecoderConfig)
        super().__init__(config)
        self.dropout = nn.Dropout(config.input_dropout)

    def preprocess(self, hidden_state: EncoderOutput) -> Tensor:
        encoder_output = super().preprocess(hidden_state)
        encoder_output *= hidden_state.attention_mask.unsqueeze(-1).float()
        seq_len = hidden_state.attention_mask.sum(dim=1, keepdim=True).float()
        pooled_output = encoder_output.sum(dim=1) / seq_len
        return self.dropout(pooled_output)


@dataclass
class LinearDecoderConfig(BaseDecoderConfig):
    num_layers: int = 1

    def build(self):
        return LinearDecoder(self)


class LinearDecoder(BaseDecoder):
    def __init__(self, config: LinearDecoderConfig):
        assert isinstance(config, LinearDecoderConfig)
        super().__init__(config)
        self.config = config
        self.mapping = nn.Linear(config.input_dim, config.output_dim)


@dataclass
class LSTMDecoderConfig(BaseDecoderConfig):
    num_layers: int = 1
    dropout: float = 0.
    bidirectional: bool = True

    def build(self):
        return LSTMDecoder(self)


class LSTMDecoder(BaseDecoder):
    def __init__(self, config: LSTMDecoderConfig):
        assert isinstance(config, LSTMDecoderConfig)
        super().__init__(config)
        self.config = config
        self.mapping = nn.LSTM(config.input_dim,
                               config.output_dim,
                               config.num_layers,
                               dropout=config.dropout,
                               bidirectional=config.bidirectional,
                               batch_first=True)


@dataclass
class BaseModelConfig:
    encoder_config: BaseEncoderConfig = BaseEncoderConfig()
    decoder_config: BaseDecoderConfig = BaseDecoderConfig()

    def build(self):
        return BaseModel(self)


class BaseModel(nn.Module):
    def __init__(self, config: BaseModelConfig):
        assert isinstance(config, BaseModelConfig)
        super().__init__()
        self.config = config
        self.evaluator = None
        self.encoder = self.config.encoder_config.build()
        decoder_config = self.config.decoder_config
        decoder_config.num_encoder_layers = self.encoder.get_num_layers()
        assert decoder_config.input_dim == -1
        decoder_config.input_dim = self.encoder.get_hidden_dim()
        if decoder_config.output_dim == -1:
            decoder_config.output_dim = self.encoder.get_hidden_dim()
        self.decoder = decoder_config.build()

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def reset_metrics(self):
        self.evaluator.reset()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        assert self.evaluator is not None
        metrics = self.evaluator.get_metric()
        if reset:
            self.reset_metrics()
        return metrics

    def embed(self, inputs: BaseInput) -> Tensor:
        assert self.evaluator is not None
        enc_out = self.encoder.encode(inputs)
        assert isinstance(enc_out, EncoderOutput)
        return self.decoder.decode(enc_out)

    def forward(self, *tensors: Tensor):
        raise NotImplementedError
