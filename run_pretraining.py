# --------------------------------------------------------
# Bootstrapped Masked Autoencoders for Vision BERT Pretraining
# By Xiaoyi Dong
# Licensed under The MIT License
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

import torch.distributed as dist
import torchvision.utils
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn

import utils
from pathlib import Path

from timm.models import create_model
from optim_factory import create_adamw_optimizer

from timm.utils import *
from datasets import build_pretraining_dataset
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import models
import copy

import random 
from timm.utils import ModelEma
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser('BootMAE pre-training script', add_help=False)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    # Model parameters
    parser.add_argument('--model', default='bootmae_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--num_mask_patches', default=120, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=60)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--second_input_size', default=224, type=int,
                        help='images input size for discrete vae')

    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr_scheduler', type=str, default='cos')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "lanczos")')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--train_set', default='train', help='train set name')
    parser.add_argument('--test_set', default='val', help='test set name')
    parser.add_argument('--data', default='imagenet', help='dataset: imagenet, imagenet-tsv, imagenet22k-tsv')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')

    parser.add_argument('--smooth-epoch', type=int, default=30, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--feature_weight', type=float, default=0)
    parser.add_argument('--mask_num', type=float, default=147)
    parser.add_argument('--weight_mask', default=False, action='store_true')
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.999, help='the start ema decay')
    parser.add_argument('--model_ema_dynamic', action='store_true', default=False)
    parser.add_argument('--resize_scale', type=float, default=0.2)

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        #decoder_depth=args.decoder_depth, 
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
    )

    return model

@torch.no_grad()
def concat_all_gather(tensor, rank, sele=False):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    #print (rank, torch.sum(tensors_gather[rank] - tensor).item(), torch.sum(tensors_gather[rank-1]-tensor).item())
    if sele:
        tensors_gather = [tensors_gather[i] for i in range(len(tensors_gather)) if i!=rank]

    output = torch.cat(tensors_gather, dim=0)
    return output

def main(args):
    #utils.init_distributed_mode(args)
    assert args.data in ['imagenet', 'imagenet-tsv', 'imagenet22k-tsv']

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        print('**********find WORLD_SIZE %d in env**********' % int(os.environ['WORLD_SIZE']))
        if args.distributed and args.num_gpu > 1:
            args.num_gpu = 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print('************World_size is %d, current rank %d ***********, local rank %d' % (args.world_size, args.rank, args.local_rank))
    assert args.rank >= 0
    

    torch.distributed.barrier()
    utils.setup_for_distributed(args.rank == 0)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get dataset
    dataset_train = build_pretraining_dataset(args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks // args.update_freq

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    model.to(device)
    model_without_ddp = model
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()

    print("Base LR = %.8f" % args.lr)
    args.lr = args.lr * total_batch_size / 256
    print("Adjuested LR = %.8f" % args.lr)
    print("Adjuested Min LR = %.8f" % args.min_lr)

    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_adamw_optimizer(
        args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    if args.lr_scheduler == 'cos':
        print ('USEING COS LR')
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
    elif args.lr_scheduler == 'step':
        print ('USEING STEP LR')
        lr_schedule_values = utils.step_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            update_freq=args.update_freq, smooth_epoch=args.smooth_epoch,
            rank=args.local_rank,
            output_dir=args.output_dir, 
            args=args, model_ema=model_ema,
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




def train_one_epoch(model: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    update_freq=None, smooth_epoch=None, 
                    lr_schedule_values=None, wd_schedule_values=None, 
                    rank=0, output_dir=None, args=None, model_ema=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()
    
    ### linear increase feature_weight at 25% epochs
    feature_weight = args.feature_weight * (1+epoch) / args.epochs
    feature_weight = min(feature_weight*4, args.feature_weight)

    ### step ema
    model_ema.decay = utils.adjust_ema_momentum(epoch, args)
    print ('Dynamic EMA DECAY ', model_ema.decay)

    win_size = args.window_size[0]
    patch_size = args.patch_size[0]
    seq_len = win_size ** 2
    pool = torch.nn.AvgPool2d(3,1,padding=1).cuda()
    mask_len = int(args.mask_num)
    LN = nn.LayerNorm(model.module.embed_dim, eps=1e-6, elementwise_affine=False).cuda()

    for data_iter_step, (imgs, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images = imgs[0].to(device, non_blocking=True)
        masks = imgs[1].to(device, non_blocking=True)
        bz, _, H, W = images.shape

        with torch.no_grad():
            idx_shuffle = utils.mask_to_index(masks)
            temp_len = mask_len

            ### index to mask
            idx_unshuffle = torch.argsort(idx_shuffle, dim=1)
            loss_mask = [0]*(seq_len-temp_len) + [1]*temp_len
            loss_mask = torch.Tensor(loss_mask).reshape(1, seq_len).cuda().repeat(bz,1)
            loss_mask = torch.gather(loss_mask, dim=1, index = idx_unshuffle)
            loss_mask = loss_mask.reshape(-1, 1, win_size, win_size).to(torch.float)
            if args.weight_mask:
                weight_mask = loss_mask * pool(loss_mask) + loss_mask
                perc_mask = weight_mask.reshape(-1, seq_len, 1)
                weight_mask = torch.nn.functional.interpolate(weight_mask, (H, W), mode='nearest')
            else:
                weight_mask = torch.nn.functional.interpolate(loss_mask, (H, W), mode='nearest')
                perc_mask = loss_mask.reshape(-1, seq_len, 1)

            loss_mask = torch.nn.functional.interpolate(loss_mask, (H, W), mode='nearest')
            vis_mask = 1 - loss_mask

            ### pixel norm
            patch = images.reshape(bz, 3, win_size, patch_size, win_size, patch_size).permute(0,2,4,1,3,5).reshape(-1,3,patch_size,patch_size)

            patch_mean = patch.mean(dim=[2,3], keepdim=True)
            im_mean = patch_mean.repeat(1,1,patch_size,patch_size).reshape(bz,win_size,win_size,3,patch_size,patch_size).permute(0,3,1,4,2,5).reshape(bz,3,H,W)

            patch_std = torch.sqrt(patch.var(dim=[2,3], keepdim=True, unbiased=False) + 1e-5)
            im_std = patch_std.repeat(1,1,patch_size,patch_size).reshape(bz,win_size,win_size,3,patch_size,patch_size).permute(0,3,1,4,2,5).reshape(bz,3,H,W)

            im_norm = (images - im_mean) / im_std

        with torch.cuda.amp.autocast():
            out = model(images, temp_len, idx_shuffle)

            Rloss = torch.mean(weight_mask * ((out[0] - im_norm)**2))/torch.mean(loss_mask)

            if feature_weight > 0:
                with torch.no_grad():
                    assert model_ema is not None
                    feat_model = model_ema.ema
                    feat_model.eval()
                    Feat_gt = feat_model.get_feature(images, temp_len, idx_shuffle)
                    assert Feat_gt.shape == out[1].shape
                Ploss = torch.mean(perc_mask * ((LN(Feat_gt) - LN(out[1]))**2))/torch.mean(loss_mask)
            else:
                Ploss = torch.zeros(1).cuda()

            loss = Rloss + Ploss * feature_weight

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("RLoss is {} PLoss is {}, stopping training".format(Rloss.item(), Ploss.item()),force=True)
            print ('out0',torch.sum(out[0]),force=True)
            print ('out1',torch.sum(out[1]),force=True)
            print ('perc',torch.sum(Feat_gt),force=True)
            sys.exit(1)
        
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        #torch.cuda.synchronize()
        if rank == 0 and it % 1000 == 0:# and False:
            with torch.no_grad():
                #patch = im_norm * im_std + im_mean
                dim = patch_size * patch_size * 3
                patch = patch.reshape(bz, win_size*win_size, dim)
                patch = torch.gather(patch, dim=1, index = idx_shuffle.unsqueeze(-1).repeat(1, 1, dim))
                patch = torch.gather(patch, dim=1, index = idx_unshuffle.unsqueeze(-1).repeat(1, 1, dim))
                patch = patch.reshape(bz,win_size,win_size,3,patch_size,patch_size).permute(0,3,1,4,2,5).reshape(bz,3,224,224)

                patch_1 = im_norm * im_std + im_mean
                out_im = out[0] * im_std #+ im_mean

                images_1 = images * vis_mask
                im_save = torch.cat([patch, patch_1, images_1, out_im], dim=-1)
                im_save = im_save[:32] * 0.5 + 0.5

            torchvision.utils.save_image(
                im_save,
                os.path.join(output_dir, 'train-%d.jpg' % it),
                padding=0,
                normalize=False)

        metric_logger.update(loss=loss_value)
        metric_logger.update(Rloss=Rloss.item())
        metric_logger.update(Ploss=Ploss.item())
        metric_logger.update(Pmean=torch.mean(torch.abs(out[1])).item())
        metric_logger.update(Pgt=torch.mean(torch.abs(Feat_gt)).item())

        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
