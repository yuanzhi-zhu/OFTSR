# -*- coding: utf-8 -*-
# Yuanzhi Zhu, 2024

import os
import sys
join = os.path.join
import gc
import logging
import torch
import numpy as np
from tqdm import tqdm
from models import create_model
from models.ema import ExponentialMovingAverage
from fm.image_datasets import ImageDataset
from fm import losses as losses
from fm import sampling as sampling
from fm.utils import (parse_args_and_config,
                      seed_everywhere,
                      save_code_snapshot,
                      save_checkpoint,
                      restore_checkpoint,
                      save_image_batch)
from fm import FM
from fm.augment import AugmentPipe
from datetime import datetime
import wandb
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# the following two lines are added according to:
# https://github.com/Lightning-AI/litgpt/issues/327#issuecomment-1664674460
# an error occurs when use torch.compile, and amp; does not slow down training when both are set to False
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def main():
    ##################          prepare config           #####################
    config = parse_args_and_config()
    # Access the parameters
    dataset_config = config.dataset
    config_ir = config.ir
    flow_config = config.fm_model
    network_config = config.network
    train_config = config.train
    sample_config = config.sample
    optim_config = config.optim
    dataset_config.in_channels = network_config.in_channels
    network_config.img_size = dataset_config.img_size
    network_config.num_classes = dataset_config.num_classes
    network_config.use_cond = flow_config.use_cond
    # network_config.in_channels = 2 * network_config.in_channels if flow_config.use_cond else network_config.in_channels
    
    ##################             Setup DDP            #####################
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    dist.init_process_group(backend="nccl", init_method='env://')
    dataset_config.batch_size = dataset_config.batch_size * dist.get_world_size()
    assert dataset_config.batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    device_type = 'cuda'
    torch.cuda.set_device(device)
    network_config.world_size = torch.cuda.device_count()
    
    # set random seed everywhere
    seed_everywhere(config.seed * dist.get_world_size() + rank)

    ##################          working paths           #####################
    if dist.get_rank() == 0:
        ### set up paths
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        if 'sr' in config_ir.degradation:
            if config_ir.degradation == 'sr_interp':
                ir_name = f'sr_{config_ir.mode}-sf{config_ir.scale_factor}'
            else:
                ir_name = f'{config_ir.degradation}-sf{config_ir.scale_factor}'
        else:
            ir_name = config_ir.degradation
        ir_name = ir_name + f'-sigmay_{config_ir.sigma_y}' if config_ir.sigma_y > 0 else ir_name
        flow_name = f't_{flow_config.flow_t_schedule}-sigma{config_ir.sigma_pertubation}'
        flow_name = flow_name + f'-no_cond' if not flow_config.use_cond else flow_name
        optim_name = f'bs{dataset_config.batch_size}-loss_{train_config.loss_type}-lr{optim_config.lr}'
        optim_name = optim_name + f'-amp_{train_config.amp_dtype}' if train_config.use_amp else optim_name
        expr_name = f'{ir_name}-{network_config.model_arch}-{flow_name}-{optim_name}-{config.expr}'
        work_path = join(config.work_dir, f'{expr_name}/{run_id}')
        ckpt_path = join(work_path, 'checkpoints')
        ckpt_meta_path = os.path.join(work_path, "checkpoints-meta", "checkpoint.pth")
        img_path = join(work_path, 'images')
        os.makedirs(work_path, exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(os.path.dirname(ckpt_meta_path), exist_ok=True)
        os.makedirs(img_path, exist_ok=True)
        save_code_snapshot(join(work_path, f'codes'))
        # convert config to dict and save it
        OmegaConf.save(config.__to_dict__(), os.path.join(work_path, 'config.yaml'))

    ##################          setup loggers           #####################
    if dist.get_rank() == 0:
        gfile_stream = open(f'{work_path}/std_{run_id}.log', 'w')
        handler = logging.StreamHandler(gfile_stream)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(filename)s - %(asctime)s - %(levelname)s --> %(message)s')
        handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.addHandler(console_handler)
        logger.setLevel('INFO')
        if config.wandb_project:
            wandb.login(key=config.wandb_key)
            wandb.init(
                dir=work_path,
                project=config.wandb_project,
                name=expr_name,
                config=config,
                entity=config.wandb_entity,
            )
        elif config.use_tensorboard:
            writer = SummaryWriter(log_dir=join('tensorboard', work_path))
        logger.info(f"working directory: {work_path}")
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    
    ##################          basic GPU info          #####################
    if dist.get_rank() == 0:
        logger.info(f'Using device: {device}; version: {str(torch.version.cuda)}')
        logger.info(f"using {network_config.world_size} GPUs!")
        logger.info(torch.cuda.get_device_name(0))
    logger.info(f"Starting rank={rank}, seed={config.seed}, world_size={dist.get_world_size()}.")

    ##################        create dataloaders        #####################
    img_dataset = ImageDataset(dataset_config, phase='train')
    logger.info(f'length of dataset: {len(img_dataset)}')
    sampler = DistributedSampler(
                    img_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=rank,
                    shuffle=True,
                    seed=(config.seed)
    )
    data_loader = DataLoader(
                    img_dataset,
                    batch_size=int(dataset_config.batch_size // dist.get_world_size()),
                    shuffle=False,
                    sampler=sampler,
                    num_workers=dataset_config.num_workers,
                    pin_memory=True,
                    drop_last=True
    )
    logger.info(f'length of dataloader: {len(data_loader)}')
    if dist.get_rank() == 0:
        val_img_dataset = ImageDataset(dataset_config, phase='val')
        val_data_loader = torch.utils.data.DataLoader(val_img_dataset,
                                                        batch_size=sample_config.num_psnr_sample,
                                                        shuffle=False,
                                                        num_workers=dataset_config.num_workers,
                                                        pin_memory=True)
    # apply data augmentation
    if dataset_config.use_aug:
        # augment_pipe only used in the training part
        # augment_pipe = AugmentPipe(p=0.12, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        augment_pipe = AugmentPipe(p=0.12, xflip=1e8, yflip=1)  # 50% xflip, 6% yflip
        # augment_pipe = AugmentPipe(p=0.12, xflip=1e8, yflip=0)  # 50% xflip
        # augment_pipe = AugmentPipe(p=0.12, rotate_int=1e8)  # 50% xflip, 50% yflip, 50% rotate 90 degree
    else:
        augment_pipe = None

    ##################      create model & optimizer    #####################
    unet = create_model(network_config)
    if network_config.zero_init:
        logger.info("initialize the model with random initialization")
    else:
        # load model from checkpoint
        missing_keys, unexpected_keys = unet.load_state_dict(torch.load(network_config.model_path, map_location='cpu'), strict=False)
        logger.info(f"loaded model from path: {network_config.model_path}")
        print(f"missing keys: {missing_keys}")
        print(f"unexpected keys: {unexpected_keys}")
    # change the input conv layer to adapt use_cond = True
    unet.change_input_conv()

    ### compile the model
    if train_config.compile:
        logger.info("compiling the model... (takes a ~minute)")
        unet = torch.compile(unet) # requires PyTorch 2.0
    unet = DDP(unet.to(device), device_ids=[rank], find_unused_parameters=True)
    logger.info("#################### Model: ####################")
    # logger.info(f'{unet}')
    logger.info(f'initialize model {network_config.model_arch}')
    ema = ExponentialMovingAverage(unet.parameters(), decay=network_config.ema_rate)
    optimizer = losses.get_optimizer(optim_config, unet.parameters())
    state = dict(optimizer=optimizer, model=unet, ema=ema, step=0)

    ##################         resume training          #####################
    # Resume training when intermediate checkpoints are detected
    if train_config.resume_from:
        assert train_config.pre_train_model == '', "no need pre_train_model for resume_from training"
        checkpoint_meta_dir_resume = os.path.join(train_config.resume_from, "checkpoints-meta", "checkpoint.pth")
        assert os.path.exists(checkpoint_meta_dir_resume), f"Checkpoint meta file {checkpoint_meta_dir_resume} does not exist"
        state = restore_checkpoint(checkpoint_meta_dir_resume, state, device_type)
        logger.info(f"Resuming training from checkpoint {checkpoint_meta_dir_resume}")
    initial_step = int(state['step'])

    ##################     load pre-trained model       #####################
    # Load pre-trained model if specified: for finetuning
    if train_config.pre_train_model:
        # only load the unet parameters
        checkpoint = torch.load(train_config.pre_train_model, map_location=device_type)
        ema.load_state_dict(checkpoint['ema'])
        ema.copy_to(unet.parameters())
        ema = ExponentialMovingAverage(unet.parameters(), decay=network_config.ema_rate)
        optim_config.warmup = 0     # no warmup for finetuning
        optimizer = losses.get_optimizer(optim_config, unet.parameters())
        state = dict(optimizer=optimizer, model=unet, ema=ema, step=0)
        flow = FM(model=unet, ema_model=ema, cfg=config, device=device)
        logger.info(f'loaded model from path: {train_config.pre_train_model}')
        del checkpoint
        torch.cuda.empty_cache()
    else:
        flow = FM(model=unet, ema_model=ema, cfg=config, device=device)
    flow.model.train()

    ##################    building sampling functions   #####################
    if train_config.validate_iters:
        sampling_fn = sampling.get_flow_sampler(flow, device=device)
        if sample_config.sample_N != 1:
            sampling_fn_n1 = sampling.get_flow_sampler(flow, device=device, use_ode_sampler='one_step')

    ##################          prepare val batch       #####################
    if dist.get_rank() == 0:
        ## prepare num_psnr_sample for validation
        val_batch, val_label_dic = next(iter(val_data_loader))
        val_label = val_label_dic.to(device) if dataset_config.num_classes is not None else None
        assert val_batch.shape[0] >= sample_config.psnr_batch_size
        y = flow.operator_val.forward(val_batch.to(device))
        yn = flow.noiser(y)
        y_LR = yn.clone()
        if config_ir.scale_factor > 1:
            yn = flow.operator_val.transpose(yn)
        x_0 = flow.noiser_pertub(yn)     # noise pertubation
        ## save HR and LR of the first num_sample
        save_val_batch = val_batch[:sample_config.num_sample].clone().mul_(0.5).add_(0.5)
        save_image_batch(save_val_batch, dataset_config.img_size, img_path, log_name=f'gt.png')
        save_x_0 = x_0[:sample_config.num_sample].clone().mul_(0.5).add_(0.5)
        save_image_batch(save_x_0, dataset_config.img_size, img_path, log_name=f'LR_input_perturb.png')
        save_yn = yn[:sample_config.num_sample].clone().mul_(0.5).add_(0.5)
        save_image_batch(save_yn, dataset_config.img_size, img_path, log_name=f'LR_input.png')
        save_y_LR = y_LR[:sample_config.num_sample].clone().mul_(0.5).add_(0.5)
        save_image_batch(save_y_LR, dataset_config.img_size, img_path, log_name=f'LR.png')
        if config.wandb_project:
            grid = make_grid(save_val_batch.clamp(0, 1), nrow=np.ceil(np.sqrt(sample_config.num_psnr_sample)).astype(int))
            wandb.log({"gt": [wandb.Image(to_pil_image(grid), caption=f'gt')]}, step=0)
            grid = make_grid(save_x_0.clamp(0, 1), nrow=np.ceil(np.sqrt(sample_config.num_psnr_sample)).astype(int))
            wandb.log({"LR_input_perturb": [wandb.Image(to_pil_image(grid), caption=f'LR_input_perturb')]}, step=0)
            grid = make_grid(save_yn.clamp(0, 1), nrow=np.ceil(np.sqrt(sample_config.num_psnr_sample)).astype(int))
            wandb.log({"LR_input": [wandb.Image(to_pil_image(grid), caption=f'LR_input')]}, step=0)
            grid = make_grid(save_y_LR.clamp(0, 1), nrow=np.ceil(np.sqrt(sample_config.num_psnr_sample)).astype(int))
            wandb.log({"LR": [wandb.Image(to_pil_image(grid), caption=f'LR')]}, step=0)

        gc.collect()
        torch.cuda.empty_cache()
    # visualize the first batch
    if dist.get_rank() == 0:
        try:
            batch, label_dic = next(data_iterator)
        except:
            data_iterator = iter(data_loader)
            batch, label_dic = next(data_iterator)
        save_image_batch(batch[:sample_config.num_sample].clone().mul_(0.5).add_(0.5), dataset_config.img_size, img_path, log_name=f'train_gt_batch.png')
    #################################################################################
    #                                 training loop                                 #
    #################################################################################
    logger.info("#################### Training Logs: ####################")
    optimize_fn = losses.optimization_manager(optim_config)
    train_loss_values = []
    if train_config.progress:
        pbar = tqdm(range(train_config.max_steps))
    else:
        pbar = range(train_config.max_steps)
    loss_dict = {}
    loss_dict["loss_fm"] = 0.0
    for global_step in pbar:
        if global_step < initial_step:
            continue
        optimizer.zero_grad()
        ##################          training step           #####################
        for _ in range(train_config.accumulation_steps):
            try:
                batch, label_dic = next(data_iterator)
            except:
                data_iterator = iter(data_loader)
                batch, label_dic = next(data_iterator)
            label = label_dic.to(device) if network_config.num_classes is not None else None
            # perform a train step
            loss = flow.train_step(batch.to(device), augment_pipe, label=label)
            loss /= train_config.accumulation_steps
            flow.amp_scaler.scale(loss).backward()
            loss_dict["loss_fm"] += loss.mean().item()
        optimize_fn(optimizer, flow.model.parameters(), step=state['step'], amp_scaler=flow.amp_scaler)
        # post train step
        state['step'] += 1
        state['ema'].update(flow.model.parameters())
        train_loss_values.append(loss_dict["loss_fm"])
        pbar.set_postfix(**logs) if train_config.progress else None

        ##################          record loss             #####################
        if global_step % train_config.record_iters == 0 and global_step != 0 and dist.get_rank() == 0:
            # record train loss
            current_lr = optimizer.param_groups[0]['lr']
            if config.wandb_project:
                wandb.log({"lr": current_lr,},step=global_step)
            # # save the training loss curve
            # np.save(os.path.join(work_path, f"loss_values"), train_loss_values)
            # record val loss
            with torch.no_grad():
                mini_bs = sample_config.psnr_batch_size
                total_val_mini_iter = (val_batch.shape[0]-1)//mini_bs + 1
                loss_dict["val_loss"] = 0.0
                for val_mini_iter in range(total_val_mini_iter):
                    val_mini_batch = val_batch[val_mini_iter * mini_bs: (val_mini_iter+1) * mini_bs]
                    val_mini_label = val_label[val_mini_iter * mini_bs: (val_mini_iter+1) * mini_bs] if dataset_config.num_classes is not None else None
                    loss_dict["val_loss"] +=  flow.train_step(val_mini_batch.to(device), augment_pipe, label=val_mini_label, mode='val') * val_mini_batch.shape[0]
                loss_dict["val_loss"] = loss_dict["val_loss"] / val_batch.shape[0]
            logger.info(f'step: --> {global_step:08d}; current lr: {current_lr:0.6f}; average loss: {np.average(train_loss_values):0.10f}; batch loss: {loss_dict["loss_fm"]:0.10f}; val batch loss: {loss_dict["val_loss"].item():0.10f}')
        
        # log the losses and misc
        logs = {key: value for key, value in loss_dict.items() if value != 0}
        # reset loss_dict for next step (all grads are accumulated and step is done)
        for key in loss_dict:
            loss_dict[key] = 0.0
        if config.wandb_project and dist.get_rank() == 0:
            wandb.log(logs, step=global_step)
        elif config.use_tensorboard and dist.get_rank() == 0:
            for key, value in logs.items():
                writer.add_scalar(key, value, global_step)
        
        ##################      save meta checkpoint        #####################
        # Save a temporary checkpoint to resume training after pre-emption periodically
        if global_step % train_config.snapshot_freq_for_preemption == 0 and global_step != 0 and dist.get_rank() == 0:
            save_checkpoint(ckpt_meta_path, state)

        ##################          save checkpoint         #####################
        if train_config.snapshot_freq and global_step % train_config.snapshot_freq == 0 and global_step != 0 and dist.get_rank() == 0:
            # Save the checkpoint.
            save_step = global_step // train_config.snapshot_freq
            save_checkpoint(os.path.join(ckpt_path, f'checkpoint_{save_step}.pth'), state, ema_only=True)
            logger.info(f"[SAVE] --> step: {global_step:08d}; save checkpoint checkpoint_{save_step}.pth")
        
        ##################         metric validation        #####################
        if train_config.validate_iters > 0 and global_step % train_config.validate_iters == 0 and global_step != 0 and dist.get_rank() == 0:
            ema.store(unet.parameters())
            unet.eval()
            ema.copy_to(unet.parameters())
            psnr, lpips_score, samples, _, nfe = flow.image_restoration(val_batch, sampling_fn)
            if sample_config.sample_N != 1:
                psnr_n1, lpips_score_n1, samples_n1, _, _ = flow.image_restoration(val_batch, sampling_fn_n1)
            else:
                psnr_n1, lpips_score_n1 = psnr, lpips_score
            ema.restore(unet.parameters())
            unet.train()
            logger.info(f"[EVAL] --> step: {global_step:08d}; nfe: {nfe:02f}; current psnr: {psnr:06f} dB; current psnr_n1: {psnr_n1:06f} dB")
            if config.wandb_project:
                wandb.log({"psnr": psnr, "psnr_n1": psnr_n1,}, step=global_step)
            elif config.use_tensorboard:
                writer.add_scalar("psnr", psnr, global_step)
                writer.add_scalar("psnr_n1", psnr_n1, global_step)
            if config_ir.calc_LPIPS:
                logger.info(f"[EVAL] --> step: {global_step:08d}; nfe: {nfe:02f};  current lpips: {lpips_score:0.6f}; current lpips_n1: {lpips_score_n1:06f}")
                if config.wandb_project:
                    wandb.log({"lpips": lpips_score, "lpips_n1": lpips_score_n1,},step=global_step)
                elif config.use_tensorboard:
                    writer.add_scalar("lpips", lpips_score, global_step)
                    writer.add_scalar("lpips_n1", lpips_score_n1, global_step)
            # save val images
            save_image_batch(samples[:sample_config.num_sample].clone().mul_(0.5).add_(0.5), dataset_config.img_size, img_path, log_name=f'iter_{str(global_step).zfill(8)}.png')
            if sample_config.sample_N != 1:
                save_image_batch(samples_n1[:sample_config.num_sample].clone().mul_(0.5).add_(0.5), dataset_config.img_size, img_path, log_name=f'iter_{str(global_step).zfill(8)}_n1.png')
            # if config.wandb_project:
            #     grid = make_grid(samples[:sample_config.num_sample].clone().mul_(0.5).add_(0.5).clamp(0, 1), nrow=np.ceil(np.sqrt(sample_config.num_psnr_sample)).astype(int))
            #     wandb.log({"samples": [wandb.Image(to_pil_image(grid), caption=f'sample_steps_{nfe}')]}, step=global_step)
            #     if sample_config.sample_N != 1:
            #         grid = make_grid(samples_n1[:sample_config.num_sample].clone().mul_(0.5).add_(0.5).clamp(0, 1), nrow=np.ceil(np.sqrt(sample_config.num_psnr_sample)).astype(int))
            #         wandb.log({"samples_n1": [wandb.Image(to_pil_image(grid), caption=f'sample_steps_1')]}, step=global_step)
    
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    sys.exit(main())
