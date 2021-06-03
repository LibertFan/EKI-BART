# from fairseq.data import BertDictionary
import os
import torch
import math
import numpy as np
import logging
import itertools

from fairseq import search, utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from fairseq.criterions import FairseqCriterion, register_criterion

logger = logging.getLogger(__name__)


@register_task('translation_segment2')
class TranslationSegment2Task(TranslationTask):
    def build_generator(self, args):
        from .sequence_generator import SequenceGenerator, SequenceGeneratorWithAlignment

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, 'sampling', False)
        sampling_topk = getattr(args, 'sampling_topk', -1)
        sampling_topp = getattr(args, 'sampling_topp', -1.0)
        diverse_beam_groups = getattr(args, 'diverse_beam_groups', -1)
        diverse_beam_strength = getattr(args, 'diverse_beam_strength', 0.5),
        match_source_len = getattr(args, 'match_source_len', False)
        diversity_rate = getattr(args, 'diversity_rate', -1)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError('Provided Search parameters are mutually exclusive.')
        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert sampling_topp < 0 or sampling, '--sampling-topp requires --sampling'

        if sampling:
            search_strategy = search.Sampling(self.target_dictionary, sampling_topk, sampling_topp)
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(self.target_dictionary, diversity_rate)
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        if getattr(args, 'print_alignment', False):
            seq_gen_cls = SequenceGeneratorWithAlignment
        else:
            seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            self.target_dictionary,
            beam_size=getattr(args, 'beam', 5),
            max_len_a=getattr(args, 'max_len_a', 0),
            max_len_b=getattr(args, 'max_len_b', 200),
            min_len=getattr(args, 'min_len', 1),
            normalize_scores=(not getattr(args, 'unnormalized', False)),
            len_penalty=getattr(args, 'lenpen', 1),
            unk_penalty=getattr(args, 'unkpen', 0),
            temperature=getattr(args, 'temperature', 1.),
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            search_strategy=search_strategy,
        )

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:b
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(os.pathsep)
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return SegmentDataset(
            src_tokens, src_lengths, self.source_dictionary,
            left_pad_source=True,  # self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False,
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)
        tgt_datasets.append(
            data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        )

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    return SegmentDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset,
        remove_token_id=src_dict.eos(),
    )


class SegmentDataset(LanguagePairDataset):
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
        append_bos=False,
        remove_token_id=None,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.append_bos = append_bos
        self.remove_token_id = remove_token_id

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        concept_padding_mask = np.clip(((src_item == self.src_dict.eos_index).cumsum(-1).cumsum(-1) >= 2), a_min=0, a_max=1)
        src_item_list = list(src_item.numpy())
        bos_position_id, eos_position_id = \
            src_item_list.index(self.src_dict.bos_index), src_item_list.index(self.src_dict.eos_index)
        position = [0] + [1] * (eos_position_id - bos_position_id - 1) + \
            list(range(2, len(src_item) - eos_position_id + 2))
        position = torch.Tensor(position).add(self.src_dict.pad()+1).long()
        chunk = []
        chunk_id = self.src_dict.pad() + 1
        for token in src_item_list:
            chunk.append(chunk_id)
            if token == self.src_dict.eos_index:
                chunk_id += 1
        chunk = torch.Tensor(chunk).long()

        label = [-1] * len(src_item_list)
        if tgt_item is not None:
            tgt_item_list = list(tgt_item.numpy())
            for idx in range(eos_position_id+1, len(src_item_list)):
                if src_item_list[idx] == self.src_dict.eos_index:
                    continue
                elif src_item_list[idx] in tgt_item_list[:-1]:
                    label[idx] = 1
                else:
                    label[idx] = 0
            label = torch.Tensor(label).long()

            n_concept = eos_position_id + 1
            prototype_in_target = [1 for _ in range(n_concept)]
            for src_token in src_item_list[n_concept:]:
                if src_token in tgt_item_list[:-1]:
                    prototype_in_target.append(1)
                else:
                    prototype_in_target.append(0)
            prototype_in_target = torch.Tensor(prototype_in_target).long()
        else:
            label = None
            prototype_in_target = None

        if self.remove_token_id is not None:
            # left_tokens = src_item.ne(self.remove_token_id)
            remove_tokens = src_item.eq(self.remove_token_id)
            remove_cumsum = remove_tokens.long().cumsum(-1)
            left_tokens = ~remove_tokens + remove_cumsum.eq(remove_cumsum.max(-1)[0])
            src_item = src_item[left_tokens]
            position = position[left_tokens]
            chunk = chunk[left_tokens]
            if label is not None:
                label = label[left_tokens]
            if prototype_in_target is not None:
                prototype_in_target = prototype_in_target[left_tokens]
            concept_padding_mask = concept_padding_mask[left_tokens]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'segment': chunk,
            'position': position,
            'chunk': chunk,
            'label': label,
            'prototype_in_target': prototype_in_target,
            'concept_padding_mask': concept_padding_mask
        }
        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        return example

    def collater(self, samples):
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, merge_pad_idx=pad_idx):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            merge_pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def expand_chunk_to_mask(chunks):
        chunk_masks = chunks.unsqueeze(1).eq(chunks.unsqueeze(2))
        concept_mask = chunks.eq(pad_idx+1)
        chunk_masks.add_(concept_mask.unsqueeze(1)).add_(concept_mask.unsqueeze(2))
        return ~chunk_masks

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    src_segments = merge('segment', left_pad=left_pad_source)
    src_positions = merge('position', left_pad=left_pad_source)
    src_chunks = merge('chunk', left_pad=left_pad_source)
    src_chunk_masks = expand_chunk_to_mask(src_chunks)
    concept_padding_mask = merge('concept_padding_mask', left_pad=left_pad_source, merge_pad_idx=1)
    if samples[0].get('label') is not None:
        src_labels = merge('label', left_pad=left_pad_source, merge_pad_idx=-1)
    else:
        src_labels = None
    if samples[0].get('prototype_in_target') is not None:
        prototype_in_target = merge('prototype_in_target', left_pad=left_pad_source, merge_pad_idx=-1)
    else:
        prototype_in_target = None

    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    src_segments = src_segments.index_select(0, sort_order)
    src_positions = src_positions.index_select(0, sort_order)
    src_chunks = src_chunks.index_select(0, sort_order)
    concept_padding_mask = concept_padding_mask.index_select(0, sort_order).bool()
    if src_labels is not None:
        src_labels = src_labels.index_select(0, sort_order)
    if prototype_in_target is not None:
        prototype_in_target = prototype_in_target.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'src_segments': src_segments,
            'src_positions': src_positions,
            'src_chunks': src_chunks,
            'src_chunk_masks': src_chunk_masks,
            'concept_padding_mask': concept_padding_mask,
        },
        'target': target,
        'src_label': src_labels
    }
    if src_labels is not None:
        batch['src_label'] = src_labels
    if prototype_in_target is not None:
        batch['net_input']['prototype_in_target'] = prototype_in_target
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    return batch
