
now=EP799_5E3_D06
MODEL=bootmae_base_patch16_224
KEY=BootMAE_Base_800

OUTPUT_DIR=OUTPUT/BootMAE/finetune/${KEY}/${now}

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
cp $0 $OUTPUT_DIR

DATA_PATH=DATASET/imagenet

FINE=OUTPUT/BootMAE/pretrain/BootMAE-bootmae_base-800/checkpoint-799.pth
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
    --model ${MODEL} --data_path $DATA_PATH \
    --input_size 224 \
    --finetune ${FINE} \
    --num_workers 8 \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 256 --lr 5e-3 --update_freq 1 \
    --warmup_epochs 20 --epochs 100 \
    --layer_decay 0.6 --backbone_decay 1 \
    --drop_path 0.1 \
    --abs_pos_emb --disable_rel_pos_bias \
    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 \
    --nb_classes 1000 --model_key model \
    --enable_deepspeed \
    --model_ema --model_ema_decay 0.9998 \
    2>&1 | tee -a ${OUTPUT_DIR}/log.txt
