#!/usr/bin/env bash

ARCH=bart9_large
USER_DIR=./src
PROBLEM=wiki

TASK=translation_segment2
ADJUST_POSITION=""
NO_SEGMENT_EMBEDDING=""
SAVE_INTERVAL_UPDATES=100
KEEP_INTERVAL_UPDATES=-1
KEEP_BEST_CHECKPOINTS=15

MAX_SEGMENTS=1
LR=5e-3
SEED=357
VERSION=1
INIT_SCALE="--init-scale 1e-2"
ADAPTOR_INIT_SCALE=1e-3
POSITION_EMBEDDING_INIT_SCALE="--position-embedding-init-scale 5e-3"
APPLY_DECODER_ENCODER_POSITION="--apply-decoder-encoder-position"
DECISION_LAMBDA=0.5
LABEL_SMOOTHING=0.1
UPDATE_FREQ=1
WARMUP_UPDATES=500
MAX_TOKENS=1024
WEIGHT_DECAY=0.01
ADAM_BETAS="(0.9,0.999)"
ADAM_EPS=1e-8
CLIP_NORM=0.1
DROPOUT=0.1
ATTENTION_DROPOUT=0.1
TOTAL_UPDATES=5000 # 10000
MAX_EPOCH=10
LOG_INTERVAL=10
FP16=""

PROBLEM=${PROBLEM}
VERSION=${VERSION}
DATA_DIR=./data/bin/${PROBLEM}-bin
SAVE_DIR=./log/${PROBLEM}/${ARCH}_v${VERSION}
BART_PATH=./log/bart/bart.large/model.pt
mkdir -p ${SAVE_DIR}

echo DATA_DIR: ${DATA_DIR}
echo SAVE_DIR: ${SAVE_DIR}
echo USER_DIR: ${USER_DIR}

if [[ ! -f "${SAVE_DIR}/checkpoint_last.pt" ]]
then
    fairseq-train ${DATA_DIR} --seed ${SEED} \
        --restore-file ${BART_PATH} --save-dir ${SAVE_DIR} --user-dir ${USER_DIR} \
        --max-tokens ${MAX_TOKENS} \
        --task ${TASK} \
        --source-lang src --target-lang tgt \
        --layernorm-embedding \
        --share-all-embeddings \
        --share-decoder-input-output-embed \
        --reset-optimizer --reset-dataloader --reset-meters \
        --required-batch-size-multiple 1 \
        --arch ${ARCH} \
        --criterion label_smoothed_cross_entropy_with_decision \
        --decision-lambda ${DECISION_LAMBDA} \
        --label-smoothing ${LABEL_SMOOTHING} \
        --dropout ${DROPOUT} --attention-dropout ${ATTENTION_DROPOUT} \
        --weight-decay ${WEIGHT_DECAY} --optimizer adam --adam-betas ${ADAM_BETAS} --adam-eps ${ADAM_EPS} \
        --clip-norm ${CLIP_NORM} \
        --lr-scheduler polynomial_decay --lr ${LR} --total-num-update $TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
        --max-epoch ${MAX_EPOCH} --max-update ${TOTAL_UPDATES} \
        ${FP16} --update-freq $UPDATE_FREQ --log-interval ${LOG_INTERVAL} --log-format simple \
        --save-interval-updates ${SAVE_INTERVAL_UPDATES} --keep-interval-updates ${KEEP_INTERVAL_UPDATES} \
        --skip-invalid-size-inputs-valid-test \
        --find-unused-parameters --adaptor-init-scale ${ADAPTOR_INIT_SCALE} \
        ${ADJUST_POSITION} ${NO_SEGMENT_EMBEDDING} ${INIT_SCALE} --max-segments ${MAX_SEGMENTS} \
        ${POSITION_EMBEDDING_INIT_SCALE} ${APPLY_DECODER_ENCODER_POSITION} --bpe gpt2 \
        | tee -a ${SAVE_DIR}/train_log.txt
else
    fairseq-train ${DATA_DIR} --seed ${SEED} --user-dir ${USER_DIR} \
        --save-dir ${SAVE_DIR} \
        --max-tokens ${MAX_TOKENS} \
        --task ${TASK} \
        --source-lang src --target-lang tgt \
        --layernorm-embedding \
        --share-all-embeddings \
        --share-decoder-input-output-embed \
        --required-batch-size-multiple 1 \
        --arch ${ARCH} \
        --criterion label_smoothed_cross_entropy_with_decision \
        --decision-lambda ${DECISION_LAMBDA} \
        --label-smoothing ${LABEL_SMOOTHING} \
        --dropout ${DROPOUT} --attention-dropout ${ATTENTION_DROPOUT} \
        --weight-decay ${WEIGHT_DECAY} --optimizer adam --adam-betas ${ADAM_BETAS} --adam-eps ${ADAM_EPS} \
        --clip-norm ${CLIP_NORM} \
        --lr-scheduler polynomial_decay --lr ${LR} --total-num-update $TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
        --max-epoch ${MAX_EPOCH} --max-update ${TOTAL_UPDATES} \
        ${FP16} --update-freq $UPDATE_FREQ --log-interval ${LOG_INTERVAL} --log-format simple \
        --save-interval-updates ${SAVE_INTERVAL_UPDATES} --keep-interval-updates ${KEEP_INTERVAL_UPDATES} \
        --skip-invalid-size-inputs-valid-test \
        --find-unused-parameters --adaptor-init-scale ${ADAPTOR_INIT_SCALE} \
        ${ADJUST_POSITION} ${NO_SEGMENT_EMBEDDING} ${INIT_SCALE} --max-segments ${MAX_SEGMENTS} \
        ${POSITION_EMBEDDING_INIT_SCALE} ${APPLY_DECODER_ENCODER_POSITION} --bpe gpt2 \
        | tee -a ${SAVE_DIR}/train_log.txt
fi

# --keep-interval-updates ${KEEP_INTERVAL_UPDATES} \
