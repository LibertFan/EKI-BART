# EKI-BART

This is an implementation for *An Enhanced Knowledge Injection Model for Commonsense Generation* in COLING 2020.


## Requirements and Installation

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
