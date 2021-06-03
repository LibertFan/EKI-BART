# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension
"""

import logging

import torch.nn as nn

from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerModel
from fairseq.models.bart import BARTModel, BARTClassificationHead
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from bart9a.hub_interface import BART3HubInterface as BARTHubInterface
from .encoder import Encoder, EncoderOut
from .decoder import Decoder

logger = logging.getLogger(__name__)


@register_model('bart9a')
class BART9AModel(TransformerModel):

    @classmethod
    def hub_models(cls):
        return {
            'bart.large': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz',
            'bart.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz',
            'bart.large.cnn': 'http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz',
        }

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        # self.apply(init_bert_params)

        self.adaptor = Adaptor(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=self.args.adaptor_inner_dim,
            activation_fn=self.args.activation_fn,
            adaptor_dropout=self.args.adaptor_dropout,
            adaptor_init_scale=self.args.adaptor_init_scale,
            apply_adaptor_segment=self.args.apply_adaptor_segment,
            max_segments=self.args.max_segments,
            padding_idx=1,  # self.args.src_padding_idx,
            segment_init_scale=self.args.adaptor_segment_init_scale,
            apply_adaptor_classification=self.args.apply_adaptor_classification,
            )

        self.classification_heads = nn.ModuleDict()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file='model.pt',
        data_name_or_path='.',
        bpe='gpt2',
        **kwargs,
    ):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return BARTHubInterface(x['args'], x['task'], x['models'][0])

    @staticmethod
    def add_args(parser):
        super(BART9AModel, BART9AModel).add_args(parser)
        parser.add_argument('--max-segments', type=int, metavar='N',
                            help='max segments', default=2)
        parser.add_argument('--adaptor-inner-dim', type=int, metavar='N',
                            help='adaptor inner dimension')
        parser.add_argument('--adaptor-dropout', type=float, metavar='D',
                            help='adaptor dropout')
        parser.add_argument('--adaptor-init-scale', type=float, metavar='D',
                            help='scale of adaptor initialization')
        parser.add_argument('--no-segment-embedding', action='store_true',
                            help='not add segment embedding', default=False)
        parser.add_argument('--adjust-position', action='store_true',
                            help='adjust position indices', default=False)
        parser.add_argument('--init-scale', type=float, metavar='D',
                            help='scale of segment embedding variance')
        parser.add_argument('--apply-chunk-mask', action='store_true',
                            help='add mask to different chunks', default=False)
        parser.add_argument('--apply-adaptor-segment', action='store_true',
                            help='apply segment into adaptor', default=False)
        parser.add_argument('--adaptor-segment-init-scale', type=float, metavar='D',
                            help='scale of adaptor segments initialization')
        parser.add_argument('--apply-adaptor-classification', action='store_true',
                            help='apply adaptor classification', default=False)
        # position_embedding_init_scale
        parser.add_argument('--position-embedding-init-scale', type=float, metavar='D',
                            help='scale of position embedding')
        # add_encoder_position
        parser.add_argument('--apply-decoder-encoder-position', action='store_true',
                            help='apply encoder position in decoder', default=False)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return Encoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return Decoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    @property
    def supported_targets(self):
        return {'self'}

    # @classmethod
    # def build_model(cls, args, task):
    #     args.src_padding_idx = task.source_dictionary.pad()
    #     # super(bart9aModel, bart9aModel).build_model(args, task)
    #     super(bart9aModel, cls).build_model(args, task)

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens,
        features_only=False, classification_head_name=None, **kwargs
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            **kwargs,
        )
        encoder_x = encoder_out.encoder_out
        encoder_x, encoder_px, scale = self.adaptor(encoder_x, **kwargs)
        encoder_out = EncoderOut(
            encoder_out=encoder_x,  # T x B x C
            encoder_padding_mask=encoder_out.encoder_padding_mask,  # B x T
            encoder_embedding=encoder_out.encoder_embedding,  # B x T x C
            encoder_states=encoder_out.encoder_states,  # List[T x B x C]
            encoder_tokens=encoder_out.encoder_tokens,
            encoder_segments=encoder_out.encoder_segments
        )
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            **kwargs,
        )

        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(self.encoder.dictionary.eos()), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            x = self.classification_heads[classification_head_name](
                sentence_representation
            )

        scale = scale.transpose(0, 1)
        return (x, extra), scale

    def forward_encoder(
        self,
        src_tokens,
        src_lengths,
        **kwargs,
    ):

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            **kwargs,
        )
        encoder_x = encoder_out.encoder_out
        encoder_x, _, _ = self.adaptor(encoder_x)
        encoder_out = EncoderOut(
            encoder_out=encoder_x,  # T x B x C
            encoder_padding_mask=encoder_out.encoder_padding_mask,  # B x T
            encoder_embedding=encoder_out.encoder_embedding,  # B x T x C
            encoder_states=encoder_out.encoder_states,  # List[T x B x C]
            encoder_tokens=encoder_out.encoder_tokens,
            encoder_segments=encoder_out.encoder_segments
        )
        return encoder_out

    def register_adaptor(self, **kwargs):
        """Register adaptor."""

        self.adaptor = Adaptor(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=self.args.adaptor_inner_dim,
            activation_fn=self.args.activation_fn,
            adaptor_dropout=self.args.adaptor_dropout,
            adaptor_init_scale=self.args.adaptor_init_scale,
            apply_adaptor_segment=self.args.apply_adaptor_segment,
            max_segments=self.args.max_segments,
            padding_idx=1,  # self.args.src_padding_idx,
            segment_init_scale=self.args.adaptor_segment_init_scale,
            apply_adaptor_classification=self.args.apply_adaptor_classification,
            )

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = BARTClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict['encoder.embed_tokens.weight'].size(0)
        if loaded_dict_size == len(self.encoder.dictionary) + 1 and '<mask>' not in self.encoder.dictionary:
            state_dict['encoder.embed_tokens.weight'] = state_dict['encoder.embed_tokens.weight'][:loaded_dict_size-1, :]
            state_dict['decoder.embed_tokens.weight'] = state_dict['decoder.embed_tokens.weight'][:loaded_dict_size-1, :]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v

        prefix = name + '.' if name != '' else ''
        for k, v in self.state_dict().items():
            if prefix + k not in state_dict:
                logger.info('Complete', prefix + k)
                state_dict[prefix + k] = v


class Adaptor(nn.Module):
    """ Adaptor for encoder with gated."""

    def __init__(
            self,
            input_dim,
            inner_dim,
            activation_fn,
            adaptor_dropout,
            adaptor_init_scale=1e-3,
            apply_adaptor_segment=False,
            max_segments=None,
            padding_idx=1,
            segment_init_scale=1e-5,
            apply_adaptor_classification=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        nn.init.normal_(self.dense.weight, std=adaptor_init_scale)
        # nn.init.xavier_normal_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=adaptor_dropout)
        self.out_proj = nn.Linear(inner_dim, input_dim)
        nn.init.normal_(self.out_proj.weight, std=adaptor_init_scale)
        nn.init.zeros_(self.out_proj.bias)
        self.out_scale = 2.0
        if apply_adaptor_segment and max_segments is not None and isinstance(max_segments, int):
            self.embed_segments = Embedding(
                max_segments+padding_idx+1,
                input_dim,
                init_scale=segment_init_scale if segment_init_scale is not None else None
            )
        else:
            self.embed_segments = None
        if apply_adaptor_classification:
            self.adaptor_classification = nn.Linear(inner_dim, 2)
            # nn.init.xavier_normal_(self.adaptor_classification.weight)
            nn.init.normal_(self.adaptor_classification.weight, std=adaptor_init_scale)
            nn.init.zeros_(self.adaptor_classification.bias)
        else:
            self.adaptor_classification = None

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        if 'src_segments' in kwargs and self.embed_segments is not None:
            x += self.embed_segments(kwargs['src_segments'].transpose(0, 1))
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        if self.adaptor_classification is not None:
            px = self.adaptor_classification(x).softmax(-1)
        else:
            px = 0.0
        scale = self.out_proj(x).sigmoid()
        out_scale = scale * self.out_scale
        x = out_scale.mul(features)
        return x, px, scale


def Embedding(num_embeddings, embedding_dim, padding_idx=None, init_scale=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5 if init_scale is None else init_scale)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


@register_model_architecture('bart9a', 'bart9a_large')
def bart9a_large_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4*1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.max_target_positions = getattr(args, 'max_target_positions', 1024)
    args.max_source_positions = getattr(args, 'max_source_positions', 1024)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', True)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)

    args.no_segment_embedding = getattr(args, 'no_segment_embedding', False)
    args.init_scale = getattr(args, 'init_scale', None)
    args.adjust_position = getattr(args, 'adjust_position', False)
    args.apply_adaptor_segment = getattr(args, 'apply_adaptor_segment', False)
    args.adaptor_segment_init_scale = getattr(args, 'adaptor_segment_init_scale', None)

    args.adaptor_inner_dim = getattr(args, 'adaptor_inner_dim', 512)
    args.adaptor_dropout = getattr(args, 'adaptor_dropout', args.dropout)
    args.adaptor_init_scale = getattr(args, 'adaptor_init_scale', 0.001)

    args.position_embedding_init_scale = getattr(args, 'position_embedding_init_scale', 0.001)
    args.apply_decoder_encoder_position = getattr(args, 'apply_decoder_encoder_position', False)