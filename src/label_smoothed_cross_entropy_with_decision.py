# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import os

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion('label_smoothed_cross_entropy_with_decision')
class LabelSmoothedCrossEntropyCriterionWithDecision(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--decision-lambda', default=0., type=float, metavar='D',
                            help='the weights of the loss of identify prototype in targets')
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, encoder_scale = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        # print('| encoder_scale: ', encoder_scale.mean(-1)[0])
        # print('| label: ', sample['net_input']['prototype_in_target'][0])

        decision_loss = self.compute_decision_loss(
            encoder_scale, sample['net_input']['prototype_in_target'], reduce=reduce)
        loss += self.args.decision_lambda * decision_loss
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        src_sample_size = sample['target'].size(0) if self.args.sentence_avg else \
            sample['net_input']['prototype_in_target'].ne(-1).sum().item()

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'decision_loss': utils.item(decision_loss.data) if reduce else decision_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'src_ntokens': sample['net_input']['prototype_in_target'].ne(-1).sum().item(),
            'sample_size': sample_size,
            'src_sample_size': src_sample_size
        }
        return loss, sample_size, logging_output

    def compute_decision_loss(self, encoder_scales, targets, reduce=True):
        decision_loss = targets.mul(encoder_scales.mean(-1).clamp(1e-10).log()).\
            add((1-targets).mul((1-encoder_scales).mean(-1).clamp(1e-10).log())).neg()
        pad_mask = targets.eq(-1)
        if pad_mask.any():
            decision_loss.masked_fill_(pad_mask, 0.)
        if reduce:
            decision_loss = decision_loss.sum()
        loss = decision_loss
        return loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        decision_loss_sum = sum(log.get('decision_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        src_ntokens = sum(log.get('src_ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        src_sample_size = sum(log.get('src_sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'nll_loss': nll_loss_sum / ntokens / math.log(2),
            'decision_loss': decision_loss_sum / src_ntokens / math.log(2),
            # 'ppl': 2**meters['nll_loss'].avg,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'src_sample_size': src_sample_size,
        }

        return agg_output
