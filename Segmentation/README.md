# ADE20k Semantic segmentation with BootMAE

## Getting started 

1. Install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library and some required packages.

```bash
bash req.sh
```

2. Install [apex](https://github.com/NVIDIA/apex) for mixed-precision training

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

3. Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k) to prepare the ADE20k dataset.


## Fine-tuning

Command format:
```
tools/dist_train.sh <CONFIG_PATH> <NUM_GPUS>  --work-dir <SAVE_PATH> --options model.pretrained=<IMAGENET_CHECKPOINT_PATH/URL>
```

For example, using a BootMAE-base backbone with UperNet:
```bash
bash tools/dist_train.sh \
    configs/bootmae/BootMAE_Base_800_PT.py 8 \
    --options model.pretrained=<MODEL_PATH>
```
note that we write all configs in one file, change the 'data path' and 'model path' before using.

## Evaluation

Command format:
```
tools/dist_test.sh  <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```

For example, evaluate a BEiT-base backbone with UperNet:
```bash
bash tools/dist_test.sh configs/bootmae/BootMAE_Base_800_PT.py \ 
    https://github.com/LightDXY/BootMAE/releases/download/v0.1.0/BootMAE_Base_800_UperNet.pth 8 --eval mIoU
```

Expected results:
```
+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 49.12 | 60.0  | 83.75 |
+--------+-------+-------+-------+
```

Multi-scale + flip (`\*_ms.py`)
```
bash tools/dist_test.sh configs/bootmae/BootMAE_Base_800_PT_ms_test.py \
    https://github.com/LightDXY/BootMAE/releases/download/v0.1.0/BootMAE_Base_800_UperNet.pth 8 --eval mIoU
```

Expected results:
```
+--------+-------+-------+------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+------+
| global | 50.0  | 60.68 | 84.12 |
+--------+-------+-------+------+
```

---

## Acknowledgment 

This code is modified from [BeiT](https://github.com/microsoft/unilm/tree/master/beit/semantic_segmentation), built using the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library, [Timm](https://github.com/rwightman/pytorch-image-models) library, the [Swin](https://github.com/microsoft/Swin-Transformer) repository, [XCiT](https://github.com/facebookresearch/xcit) and the [SETR](https://github.com/fudan-zvg/SETR) repository.
