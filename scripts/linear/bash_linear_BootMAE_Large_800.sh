#!/bin/bash
GPUS=`nvidia-smi -L | wc -l`

RANK=0
NODE_COUNT=1
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
now=$(date +"%Y%m%d_%H%M%S")

DATA_PATH=DATASET/imagenet/
DATA=imagenet

MODEL=BootMAE_Large_800
FINETUNE=OUTPUT/BootMAE/pretrain/BootMAE-bootmae_large-800/checkpoint-799.pth

LAYER=18
OUTPUT_DIR=OUTPUT/BootMAE/linear/${MODEL}/Layer_${LAYER}

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
cp $0 $OUTPUT_DIR

python -m torch.distributed.launch --nnodes ${NODE_COUNT} \
        --node_rank ${RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        --nproc_per_node ${GPUS} \
        main_linprobe.py \
        --batch_size 1024 --accum_iter 2 \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} \
        --model large_patch16_224 --depth ${LAYER} \
        --finetune ${FINETUNE} \
        --global_pool \
        --epochs 50 \
        --blr 0.07 \
        --weight_decay 0.0 \
        --dist_eval \
        2>&1 | tee -a ${OUTPUT_DIR}/log.txt
