import os
import glob
from pathlib import Path
from argparse import ArgumentParser
import torch

from fairseq.utils import import_user_module

parser = ArgumentParser()
parser.add_argument('--user_dir', type=str, default='./bart9a', help='user_dir for fairseq')
parser.add_argument('--model_problem', type=str, default='all1v2_attach1', help='problem for evaluation')
parser.add_argument('--problem', type=str, default='all1v2_attach1', help='problem for evaluation')
parser.add_argument('--search_tag', type=str, default='', help='search tag')
parser.add_argument('--task', type=str, default='translation_segment2', help='task')
parser.add_argument('--arch', type=str, help='model architecture for training', default='bart9_large')
parser.add_argument('--gpu', type=str, help='gpu for training', default=0)
parser.add_argument('--version', type=str, help='model version for evaluation', default=1)
parser.add_argument('--lenpen', type=float, default=0.0)
parser.add_argument('--beam', type=int, default=5)
parser.add_argument('--minlen', type=int, default=3)
parser.add_argument('--maxlen', type=int, default=32)
parser.add_argument('--ngram', type=int, default=3)
parser.add_argument('--bsz', type=int, default=256)
parser.add_argument('--fp16', action='store_true', default=False)
args = parser.parse_args()

import_user_module(args)

# from fairseq.models import bart_adaptor as BARTModel
from bart9a.model import BART9AModel as BARTModel

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
print("| VISIBLE_DEVICES: ", os.environ['CUDA_VISIBLE_DEVICES'])
dev_res_len = 4018
test_res_len = 6042


def search_results(result_dir):
    print('| result_dir: ', result_dir)
    print(os.listdir(result_dir))

    dev_res_list = glob.glob(os.path.join(result_dir, "*.valid.hypo"))
    dev_res_list = [dev_res for dev_res in dev_res_list
                    if len(open(dev_res, 'r').readlines()) == dev_res_len]
    test_res_list = glob.glob(os.path.join(result_dir, "*.test.hypo"))
    test_res_list = [test_res for test_res in test_res_list
                     if len(open(test_res, 'r').readlines()) == test_res_len]

    if (not dev_res_list) or (not test_res_list):
        return []
    both_set = set([Path(dev_res).stem.split('.')[0] for dev_res in dev_res_list]
                   ) & set([Path(test_res).stem.split('.')[0] for test_res in test_res_list])
    if both_set:
        return list(both_set)
    else:
        return []


def search_checkpoints(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "checkpoint*.pt"))
    both_set = set([Path(fn).stem.split('.')[0] for fn in fn_model_list])
    if both_set:
        return list(both_set)
    else:
        return []


arch = args.arch
version = args.version

model_dir = "./log/{}/{}_v{}".format(args.model_problem, arch, version)
search_dir = os.path.join(
    model_dir, 'Search{}'.format(args.search_tag),
    'beam{}_lenpen{}_minlen{}_maxlen{}_ngram{}'.\
        format(args.beam, args.lenpen, args.minlen, args.maxlen, args.ngram))
os.makedirs(search_dir, exist_ok=True)

ckpts = search_checkpoints(model_dir)
finished_ckpts = search_results(search_dir)
ckpts = list(set(ckpts) - set(finished_ckpts))

print('| ckpts: ', ckpts)
print('| finished_ckpts: ', finished_ckpts)
print('| to be evaluated checkpoints: ', ckpts)

data_path = "data/attach/{}".format(args.problem)
data_bin_path = "data/bin/{}-bin".format(args.problem)


for ckpt in ckpts:
    print('| {} in proc.'.format(ckpt))
    # ckpt_id = ckpt.split('.')[0].split('checkpoint')[-1]

    try:
        bart = BARTModel.from_pretrained(
            model_dir,
            checkpoint_file=ckpt+'.pt' if not ckpt.endswith('.pt') else ckpt,
            data_name_or_path=data_bin_path,
            task=args.task
        )
    except:
        print('| BART Loading Error.')
        continue

    bart.cuda()
    bart.eval()
    if args.fp16:
        print('| FP16 is employed.')
        bart.half()
    count = 1
    bsz = args.bsz

    def encode(sentence: str, addl_sentences: list = None, no_separator=True) -> torch.LongTensor:
        tokens = bart.bpe.encode(sentence)
        if len(tokens.split(' ')) > bart.max_positions - 2:
            tokens = ' '.join(tokens.split(' ')[:bart.max_positions - 2])
        bpe_sentence = '<s> ' + tokens + ' </s>'
        if addl_sentences is not None:
            for s in addl_sentences:
                bpe_sentence += ' </s>' if not no_separator else ''
                bpe_sentence += ' ' + bart.bpe.encode(s) + ' </s>'
        tokens = bart.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    for data_split in ['test', 'valid']:
        n_attach = int(args.problem.split('attach')[-1])
        concepts = [
            line.strip() for line in
            open('./data/attach/{}/{}.concept.src'.format(args.problem, data_split), 'r', encoding='utf-8').readlines()
        ]
        exemplars = []
        if n_attach > 0:
            for i in range(1, n_attach+1):
                exemplar = [
                    line.strip() for line in
                    open('./data/attach/{}/{}.exemplar.{}.src'.format(args.problem, data_split, i), 'r', encoding='utf-8').readlines()
                ]
                exemplars.append(exemplar)
            exemplars = list(zip(*exemplars))

        with open(os.path.join(search_dir, '{}.{}.hypo'.format(ckpt, data_split)), 'w', encoding='utf-8') as fout:
                for start_idx in range(0, len(concepts), args.bsz):
                    end_idx = start_idx + bsz
                    concept_batch = concepts[start_idx: end_idx]
                    # print('| concept_batch', concept_batch[0])

                    import time
                    time.sleep(10)
                    with torch.no_grad():
                        if len(exemplars) > 0:
                            # print('| exemplars_batch: ', exemplars[start_idx])
                            exemplars_batch = exemplars[start_idx: end_idx]
                            net_input = [encode(concept, exemplars)
                                         for concept, exemplars in zip(concept_batch, exemplars_batch)]
                        else:
                            net_input = [encode(concept) for concept in concept_batch]

                        hypos = bart.generate(net_input,
                                              beam=args.beam,
                                              lenpen=args.lenpen,
                                              max_len_b=args.maxlen,
                                              min_len=args.minlen,
                                              no_repeat_ngram_size=args.ngram
                                              )
                        hypotheses_batch = [bart.decode(x['tokens']) for x in hypos]

                    for hypothesis in hypotheses_batch:
                        fout.write(hypothesis + '\n')
                        fout.flush()
        fout.close()
