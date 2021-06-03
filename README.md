# EKI-BART

This is an implementation for *An Enhanced Knowledge Injection Model for Commonsense Generation* ([Paper](https://arxiv.org/abs/2012.00366)) in COLING 2020.

## Requirements and Installation

```
pip install --user fairseq==0.9.0
```

## Data Download
Please refer to the github [repo](https://github.com/INK-USC/CommonGen) for data donwloading and some baselines. 

We also provide our retrieved and preprocessed data in the [BaiduDisk](https://pan.baidu.com/s/1tLjF0kvPcxfdSG720TzpRQ) with code 36iv.
In detail, all1v4 is retrieved from in-domain dataset, and wikiv4 is retrieved from out-of-domain dataset (wiki). 
You should download the *data* from BaiduDisk and put them in this main directory.

## Running

Train the model.
```
bash src/train2.sh
```

Evaluate the model.
```
python ./src/generate2.py --task translation_segment2
    --search_tag '' 
    --arch bart9a_large
    --model_problem wiki
    --problem wiki
    --version 1 
    --gpu 0 
    --bsz 12
    --lenpen 0.0 
    --beam 5 
    --maxlen 24 
    --minlen 3 
    --ngram 3 
```
