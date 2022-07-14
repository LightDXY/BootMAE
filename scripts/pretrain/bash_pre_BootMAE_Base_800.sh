#!/bin/bash
GPUS=`nvidia-smi -L | wc -l`

now=$(date +"%Y%m%d_%H%M%S")

now=800
KEY=BootMAE
MODEL=bootmae_base
OUTPUT_DIR=OUTPUT/BootMAE/pretrain/${KEY}-${MODEL}-${now}

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
cp $0 $OUTPUT_DIR

DATA_PATH=DATASET/imagenet/

RANK=0
NODE_COUNT=1
MASTER_ADDR=127.0.0.1
MASTER_PORT=12345

echo "rank: ${RANK}"
echo "node count: ${NODE_COUNT}"
echo "master addr: ${MASTER_ADDR}"
echo "master port: ${MASTER_PORT}"

echo "run command: python -m torch.distributed.launch --nnodes ${NODE_COUNT} --node_rank ${RANK} --master_addr ${MASTER_ADDR}  --master_port ${MASTER_PORT} --nproc_per_node ${GPUS} main.py $@"

python -m torch.distributed.launch --nnodes ${NODE_COUNT} \
        --node_rank ${RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        --nproc_per_node ${GPUS} \
        run_pretraining.py \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --model ${MODEL} \
        --model_ema --model_ema_decay 0.999 --model_ema_dynamic \
        --batch_size 256 --lr 1.5e-4 --min_lr 1e-4 \
        --epochs 801 --warmup_epochs 40 --update_freq 1 \
        --mask_num 147 --feature_weight 1 --weight_mask \
        2>&1 | tee -a ${OUTPUT_DIR}/log-${RANK}.txt
