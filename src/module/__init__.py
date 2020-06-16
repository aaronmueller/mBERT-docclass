import argparse

from .component import (BaseDecoderConfig, BaseEncoderConfig,
                        BertEncoderConfig, LinearDecoderConfig,
                        LSTMDecoderConfig, MeanPoolDecoderConfig,
                        PoolDecoderConfig, XLMEncoderConfig)
from .model import ParsingConfig, SeqClassConfig, SeqLabelConfig, SeqSepTopClassConfig, SeqSepTopLabelConfig, BaselineSeqClassConfig, BaselineSeqLabelConfig


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class EncoderConfigFactory(object):

    ENCODER = {'bert': BertEncoderConfig, 'xlm': XLMEncoderConfig}

    @staticmethod
    def build_args(args: argparse.ArgumentParser) -> argparse.ArgumentParser:
        args.add_argument('--encoder',
                          choices=EncoderConfigFactory.ENCODER.keys(),
                          required=True)
        group_args = args.add_argument_group('encoder')
        add = group_args.add_argument
        add('--freeze-layer',
            type=int,
            default=BaseEncoderConfig().freeze_layer,
            help='Freeze bottom X layers (-1 => None)')
        # BERT
        add('--bert-mode', type=str, default=BertEncoderConfig().mode)
        add('--xlm-mode', type=str, default=XLMEncoderConfig().mode)
        return args

    @staticmethod
    def build_config(opt: argparse.Namespace) -> BaseEncoderConfig:
        assert opt.encoder in EncoderConfigFactory.ENCODER
        if opt.encoder == 'bert':
            return BertEncoderConfig(freeze_layer=opt.freeze_layer,
                                     mode=opt.bert_mode)
        elif opt.encoder == 'xlm':
            return XLMEncoderConfig(freeze_layer=opt.freeze_layer,
                                    mode=opt.xlm_mode)
        else:
            raise ValueError(opt.encoder)


class DecoderConfigFactory(object):

    DECODER = {
        'base': BaseDecoderConfig,
        'pool': PoolDecoderConfig,
        'meanpool': MeanPoolDecoderConfig,
        'linear': LinearDecoderConfig,
        'lstm': LSTMDecoderConfig
    }

    @staticmethod
    def build_args(args: argparse.ArgumentParser) -> argparse.ArgumentParser:
        args.add_argument('--decoder',
                          choices=DecoderConfigFactory.DECODER.keys(),
                          required=True)
        group_args = args.add_argument_group('decoder')
        add = group_args.add_argument
        add('--input-dropout',
            type=float,
            default=BaseDecoderConfig().input_dropout)
        add('--feature-layer',
            type=int,
            default=BaseDecoderConfig().feature_layer)
        add('--weighted-feature',
            type=str2bool,
            default=BaseDecoderConfig().weighted_feature)
        # LINEAR
        add('--decoder-output-dim',
            type=int,
            default=LinearDecoderConfig().output_dim)
        add('--decoder-num-layers',
            type=int,
            default=LinearDecoderConfig().num_layers)
        # LSTM
        add('--decoder-bilstm',
            type=str2bool,
            default=LSTMDecoderConfig().bidirectional)
        add('--decoder-dropout',
            type=float,
            default=LSTMDecoderConfig().dropout)
        return args

    @staticmethod
    def build_config(opt: argparse.Namespace) -> BaseDecoderConfig:
        assert opt.decoder in DecoderConfigFactory.DECODER
        if opt.decoder in ['base', 'pool', 'meanpool']:
            decoder = DecoderConfigFactory.DECODER[opt.decoder]
            return decoder(feature_layer=opt.feature_layer,
                           weighted_feature=opt.weighted_feature)
        elif opt.decoder == 'linear':
            return LinearDecoderConfig(feature_layer=opt.feature_layer,
                                       weighted_feature=opt.weighted_feature,
                                       output_dim=opt.decoder_output_dim,
                                       num_layers=opt.decoder_num_layers)
        elif opt.decoder == 'lstm':
            return LSTMDecoderConfig(feature_layer=opt.feature_layer,
                                     weighted_feature=opt.weighted_feature,
                                     output_dim=opt.decoder_output_dim,
                                     num_layers=opt.decoder_num_layers,
                                     dropout=opt.decoder_dropout,
                                     bidirectional=opt.decoder_bilstm)
        else:
            raise ValueError(opt.decoder)
