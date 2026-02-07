# -*- coding: utf-8 -*-
# Yuanzhi Zhu, 2023

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
import torch.nn.functional as F
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
import copy
from rich.pretty import pretty_repr
from omegaconf import OmegaConf
from cleanfid import fid
from torchvision.utils import save_image

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

# ----------------------------------------------- utils GAN -----------------------------------------------
DISCIMINATOR_NOISE_LEVEL = [0.1, 0.25, 0.5, 0.75]
def get_timesteps_gan(gan_noise_level, bsz, device):
    # Sample timesteps for the GAN
    if gan_noise_level == 'discrete':
        DISCIMINATOR_NOISE_LEVEL_tensor = torch.tensor(DISCIMINATOR_NOISE_LEVEL)
        timesteps_gan = DISCIMINATOR_NOISE_LEVEL_tensor[torch.randint(len(DISCIMINATOR_NOISE_LEVEL),(bsz,))]
        timesteps_gan = timesteps_gan.to(device)
    elif 'continuous_' in gan_noise_level:
        trunk_tmin, trunk_tmax = map(int, gan_noise_level.split('_')[-1].split('-'))
        timesteps_gan = torch.randint(trunk_tmin, trunk_tmax, (bsz,), device=device, dtype=torch.long)
    return timesteps_gan

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
    boot_config = config.boot
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
        solver_name = f'rk2_{config.boot.rk2_r}' if config.boot.teacher_solver == 'rk2' else config.boot.teacher_solver
        distil_name = f'distil-{config.boot.distil_loss}-solver_{solver_name}-dt{config.boot.t_step}-w_distil_{config.boot.lambda_distil}-w_bound_{config.boot.lambda_boundary}-w_align_{config.boot.lambda_align}'
        expr_name = f'{ir_name}-{network_config.model_arch}-{flow_name}-{optim_name}-{distil_name}-{gan_name}-{config.expr}'
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

    ##################     load pre-trained model       #####################
    # Load pre-trained model if specified: for finetuning
    assert train_config.pre_train_model, 'need pre train model for distillation'
    if train_config.pre_train_model:
        # only load the unet parameters
        checkpoint = torch.load(train_config.pre_train_model, map_location=device_type)
        ema.load_state_dict(checkpoint['ema'])
        ema.copy_to(unet.parameters())
        ema = ExponentialMovingAverage(unet.parameters(), decay=network_config.ema_rate)
        # optim_config.warmup = 0     # no warmup for finetuning
        optimizer = losses.get_optimizer(optim_config, unet.parameters())
        state = dict(optimizer=optimizer, model=unet, ema=ema, step=0)
        flow = FM(model=unet, ema_model=ema, cfg=config, device=device)
        flow.teacher = copy.deepcopy(unet)
        flow.teacher.requires_grad_(False)
        logger.info(f'loaded model from path: {train_config.pre_train_model}')
        del checkpoint
        torch.cuda.empty_cache()
    else:
        flow = FM(model=unet, ema_model=ema, cfg=config, device=device)

    ##################         resume training          #####################
    # Resume training when intermediate checkpoints are detected
    if train_config.resume_from:
        # assert train_config.pre_train_model == '', "no need pre_train_model for resume_from training"
        checkpoint_meta_dir_resume = os.path.join(train_config.resume_from, "checkpoints-meta", "checkpoint.pth")
        assert os.path.exists(checkpoint_meta_dir_resume), f"Checkpoint meta file {checkpoint_meta_dir_resume} does not exist"
        state = restore_checkpoint(checkpoint_meta_dir_resume, state, device_type)
        logger.info(f"Resuming training from checkpoint {checkpoint_meta_dir_resume}")
    initial_step = int(state['step'])

    flow.model.train()

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
        x_0 = flow.noiser_pertub(yn).to(device)     # noise pertubation
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
        ## test teacher and student model initialization
        with torch.no_grad():
            flow.cond = yn[:sample_config.num_sample].detach().clone().to(device)
            v_teacher = flow.model_forward_wrapper(flow.teacher, x_0[:sample_config.num_sample], torch.ones(flow.cond.shape[0], device=x_0.device) * flow.eps)
            x1_teacher = x_0[:sample_config.num_sample] + (flow.T - flow.eps) * v_teacher
            save_x1_teacher = x1_teacher.clone().mul_(0.5).add_(0.5)
            save_image_batch(save_x1_teacher, dataset_config.img_size, img_path, log_name=f'x1_teacher_t0.png')
            v_model = flow.model_forward_wrapper(flow.model, x_0[:sample_config.num_sample], torch.ones(flow.cond.shape[0], device=x_0.device) * flow.eps)
            x1_model = x_0[:sample_config.num_sample] + (flow.T - flow.eps) * v_model
            save_x1_model = x1_model.clone().mul_(0.5).add_(0.5)
            save_image_batch(save_x1_model, dataset_config.img_size, img_path, log_name=f'x1_model_t0.png')
            del v_teacher, x1_teacher, v_model, x1_model
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
    loss_dict["loss_generator"] = 0.0
    # boot loss
    loss_dict["loss_boot"] = 0.0
    loss_dict["loss_boundary"] = 0.0
    loss_dict["loss_align"] = 0.0
    # GAN loss
    loss_dict["loss_gan_g"] = 0.0
    loss_dict["loss_gan_d"] = 0.0

    if config_ir.calc_FID and dist.get_rank() == 0:
        mode="legacy_tensorflow"
        model_name="inception_v3"
        num_workers=12
        batch_size=1
        custom_feat_extractor=None
        verbose=True
        custom_image_transform=None
        custom_fn_resize=None
        use_dataparallel=True
        # build the feature extractor based on the mode and the model to be used
        if custom_feat_extractor is None and model_name=="inception_v3":
            feat_model = fid.build_feature_extractor(mode, device_type, use_dataparallel=use_dataparallel)
        
        # get all inception features for the first folder
        fdir1 = dataset_config.val_path
        fbname1 = os.path.basename(fdir1)
        np_feats1 = fid.get_folder_features(fdir1, feat_model, num_workers=num_workers,
                                        batch_size=batch_size, device=device_type, mode=mode,
                                        description=f"FID {fbname1} : ", verbose=verbose,
                                        custom_image_tranform=custom_image_transform,
                                        custom_fn_resize=custom_fn_resize)
        mu1 = np.mean(np_feats1, axis=0)
        sigma1 = np.cov(np_feats1, rowvar=False)

    for global_step in pbar:
        if global_step < initial_step:
            continue
        optimizer.zero_grad()
        # ---------------------- generator step ----------------------
        for _ in range(train_config.accumulation_steps):
            try:
                batch, label_dic = next(data_iterator)
            except:
                data_iterator = iter(data_loader)
                batch, label_dic = next(data_iterator)
            label = label_dic.to(device) if network_config.num_classes is not None else None
            # perform a train step
            # 0. prepare HR and LR
            batch, augment_labels = augment_pipe(batch) if augment_pipe is not None else (batch, None)
            # Forward measurement model y = (Ax + n)
            y = flow.operator.forward(batch)
            yn_ = flow.noiser(y)
            # transpose (upsample) if the sf > 1
            if flow.config_ir.scale_factor > 1:
                yn_ = flow.operator.transpose(yn_)
            flow.cond = yn_.detach().clone().to(device)
            yn = flow.noiser_pertub(yn_)    # noise pertubation: distribution \pi(x0)
            batch = batch.to(device)        # data distribution \pi(x1)
            # 1. x0, x1, cond, t, s, labmda
            x_0 = yn.to(device)
            x_1 = batch.to(device)
            t = torch.rand((x_1.shape[0],), device=x_1.device) * (boot_config.t_max - boot_config.t_min) + boot_config.t_min
            s = torch.clamp(t + boot_config.t_step, max=boot_config.t_max)
            t_step = s - t
            t_min = torch.ones_like(t) * boot_config.t_min
            if boot_config.lambda_distil > 0 or boot_config.lambda_align > 0:
                # 2. generate the model predictions
                v_pred_t = flow.model_forward_wrapper(flow.model, x_0, t)
                v_pred_s = flow.model_forward_wrapper(flow.model, x_0, s)
                # 3. noisy sample
                x1_t = x_0 + (flow.T - flow.eps) * v_pred_t.detach()    # detach the v_pred_t for stability
                x1_s = x_0 + (flow.T - flow.eps) * v_pred_s
                xt = x_0 + torch.einsum('b,bijk->bijk', t, v_pred_t.detach())
                # xt = torch.einsum('b,bijk->bijk', t, x1_t.detach()) + torch.einsum('b,bijk->bijk', (1 - t), x_0)
                # 4. generate the teacher predictions
                with torch.no_grad():
                    v_teacher_t = flow.model_forward_wrapper(flow.teacher, xt, t)
                    if boot_config.teacher_solver == 'euler':
                        pass
                    elif boot_config.teacher_solver == 'rk2':
                        # r == 1 --> Heun's method; r == 0.5 --> midpoint method; r == 2/3 --> Ralston's method
                        pred = v_teacher_t.clone()
                        x_2 = xt + boot_config.rk2_r * torch.einsum('b,bijk->bijk', t_step, pred)
                        t_2 = boot_config.rk2_r * s + (1-boot_config.rk2_r) * t
                        pred_2 = flow.model_forward_wrapper(flow.teacher, x_2, t_2)
                        v_teacher_t = (pred + 1/(2*boot_config.rk2_r) * (pred_2 - pred))
            # 5. calculate losses
            # 5.1: signal-ODE loss
            # the ode is: v_\theta(x0, s) - v_\theta(x0, t) = (s-t) / s * (v_\psi(xt,t) - v_\theta(x0, t))
            if boot_config.lambda_distil > 0:
                if boot_config.distil_loss == 'boot':
                    labmda_t = (1 - (t*(1-s)) / (s*(1-t))) / (t_step+1e-6)
                    delta_lambda = (labmda_t * t_step)[..., None, None, None]
                    x1_teacher_t = xt + torch.einsum('b,bijk->bijk', t_step, v_teacher_t)
                    loss_distil = (1 / (delta_lambda)**2 * F.mse_loss(x1_s, (x1_t.detach() + delta_lambda*(x1_teacher_t - x1_t.detach())).detach(), reduction='none')).mean()
                elif boot_config.distil_loss == 'v_boot':
                    # x1_s - x1_t = delta_lambda * (x0 + v_\theta(xt) - x1_t)
                    # loss_distil = F.mse_loss(x1_s, (x1_t + delta_lambda*(x_0 + (flow.T-flow.eps)*v_teacher_t - x1_t)).detach(), reduction='mean')
                    delta_lambda = (t_step / s)[..., None, None, None]
                    loss_distil = F.mse_loss(v_pred_s, (v_pred_t + delta_lambda*(v_teacher_t - v_pred_t)).detach(), reduction='mean')
                elif boot_config.distil_loss == 'pinn':
                    # loss_distil = F.mse_loss((x1_s - x1_t)/delta_lambda + x1_t, (x_0 + (flow.T - flow.eps)*v_teacher_t).detach(), reduction='mean')
                    delta_lambda = ((t_step+1e-6) / s)[..., None, None, None]
                    loss_distil = F.mse_loss((v_pred_s - v_pred_t)/delta_lambda + v_pred_t, v_teacher_t.detach(), reduction='mean')
                else:
                    raise NotImplementedError(f"distil_loss: {boot_config.distil_loss} is not implemented")
                loss_dict["loss_boot"] += loss_distil.mean().item() / train_config.accumulation_steps
            else:
                loss_distil = torch.tensor(0.0, device=device)
            # 5.2: boundary loss --> a special case of the alignment loss
            if boot_config.lambda_boundary > 0:
                v_pred_t_min = flow.model_forward_wrapper(flow.model, x_0, t_min)
                with torch.no_grad():
                    v_teacher_t_min = flow.model_forward_wrapper(flow.teacher, x_0, t_min)
                loss_boundary = F.mse_loss(v_pred_t_min, v_teacher_t_min, reduction='mean')
                loss_dict["loss_boundary"] += loss_boundary.mean().item() / train_config.accumulation_steps
            else:
                loss_boundary = torch.tensor(0.0, device=device)
            # 5.3: alignment loss
            # student input x0 and t, output velocity point to the x1 at timestep t
            # teacher input xt and t, output velocity point to the x1 at timestep t
            if boot_config.lambda_align > 0:
                loss_v_align = (v_pred_t - v_teacher_t).pow(2).view(v_pred_t.shape[0], -1).mean(dim=1)
                # x1_t = xt + (1-t) * v --> l2(x1_t, x1_teacher_t) = (1-t)^2 * l2(vt, v_teacher_t)
                loss_v_align = ((1-t)**2 * loss_v_align).mean()
                loss_dict["loss_align"] += loss_v_align.mean().item() / train_config.accumulation_steps
            else:
                loss_v_align = torch.tensor(0.0, device=device)
            # 6. total loss
            loss = boot_config.lambda_distil * loss_distil + boot_config.lambda_boundary * loss_boundary + boot_config.lambda_align * loss_v_align
            loss = loss / train_config.accumulation_steps
            flow.amp_scaler.scale(loss).backward()
            loss_dict["loss_generator"] += loss.mean().item()
        optimize_fn(optimizer, flow.model.parameters(), step=state['step'], amp_scaler=flow.amp_scaler)
        
        # post train step
        state['step'] += 1
        state['ema'].update(flow.model.parameters())
        train_loss_values.append(loss_dict["loss_generator"])
        # pbar.set_description(f"loss: {loss_dict["loss_generator"]:0.6f}")
        pbar.set_postfix(**logs) if train_config.progress else None
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

        ##################          record loss             #####################
        if global_step % train_config.record_iters == 0 and dist.get_rank() == 0:
            # record train loss
            current_lr = optimizer.param_groups[0]['lr']
            if config.wandb_project:
                wandb.log({"lr": current_lr,},step=global_step)
            elif config.use_tensorboard:
                writer.add_scalar("lr", current_lr, global_step)
            logger.info(f'step: --> {global_step:08d}; current lr: {current_lr:0.6f}; average loss: {np.average(train_loss_values):0.10f}')
            logger.info(pretty_repr(logs))
        
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
        if train_config.validate_iters > 0 and global_step % train_config.validate_iters == 0 and dist.get_rank() == 0:
            ema.store(unet.parameters())
            unet.eval()
            ema.copy_to(unet.parameters())
            psnr_t_start, lpips_score_t_start, samples_t_start, _, _ = flow.image_restoration(val_batch, t=boot_config.t_min)
            psnr_t_mid, lpips_score_t_mid, samples_t_mid, _, _ = flow.image_restoration(val_batch, t=0.5)
            psnr_t_end, lpips_score_t_end, samples_t_end, _, _ = flow.image_restoration(val_batch, t=boot_config.t_max)
            ema.restore(unet.parameters())
            unet.train()
            logger.info(f"[EVAL] --> step: {global_step:08d}; psnr_t_0: {psnr_t_start:0.6f}; psnr_t_mid: {psnr_t_mid:0.6f}; psnr_t_end: {psnr_t_end:0.6f}")
            if config.wandb_project:
                wandb.log({"psnr_t_0": psnr_t_start, "psnr_t_mid": psnr_t_mid, "psnr_t_end": psnr_t_end},step=global_step)
            elif config.use_tensorboard:
                writer.add_scalar("psnr_t_0", psnr_t_start, global_step)
                writer.add_scalar("psnr_t_mid", psnr_t_mid, global_step)
                writer.add_scalar("psnr_t_end", psnr_t_end, global_step)
            if config_ir.calc_LPIPS:
                logger.info(f"[EVAL] --> step: {global_step:08d}; lpips_t_0: {lpips_score_t_start:0.6f}; lpips_t_mid: {lpips_score_t_mid:0.6f}; lpips_t_end: {lpips_score_t_end:0.6f}")
                if config.wandb_project:
                    wandb.log({"lpips_t_0": lpips_score_t_start, "lpips_t_mid": lpips_score_t_mid, "lpips_t_end": lpips_score_t_end},step=global_step)
                elif config.use_tensorboard:
                    writer.add_scalar("lpips_t_0", lpips_score_t_start, global_step)
                    writer.add_scalar("lpips_t_mid", lpips_score_t_mid, global_step)
                    writer.add_scalar("lpips_t_end", lpips_score_t_end, global_step)
            # save val images
            save_image_batch(samples_t_start[:sample_config.num_sample].clone().mul_(0.5).add_(0.5), dataset_config.img_size, img_path, log_name=f'iter_{str(global_step).zfill(8)}_t_0.png')
            save_image_batch(samples_t_mid[:sample_config.num_sample].clone().mul_(0.5).add_(0.5), dataset_config.img_size, img_path, log_name=f'iter_{str(global_step).zfill(8)}_t_mid.png')
            save_image_batch(samples_t_end[:sample_config.num_sample].clone().mul_(0.5).add_(0.5), dataset_config.img_size, img_path, log_name=f'iter_{str(global_step).zfill(8)}_t_end.png')
            # if config.wandb_project:
            #     grid = make_grid(samples[:sample_config.num_sample].clone().mul_(0.5).add_(0.5).clamp(0, 1), nrow=np.ceil(np.sqrt(sample_config.num_psnr_sample)).astype(int))
            #     wandb.log({"samples": [wandb.Image(to_pil_image(grid), caption=f'sample_steps_{nfe}')]}, step=global_step)
            #     if sample_config.sample_N != 1:
            #         grid = make_grid(samples_n1[:sample_config.num_sample].clone().mul_(0.5).add_(0.5).clamp(0, 1), nrow=np.ceil(np.sqrt(sample_config.num_psnr_sample)).astype(int))
            #         wandb.log({"samples_n1": [wandb.Image(to_pil_image(grid), caption=f'sample_steps_1')]}, step=global_step)
            if config_ir.calc_FID:
                # save the images for FID calculation
                os.makedirs(os.path.join(img_path, 'fids'), exist_ok=True)
                for img_idx in range(val_batch.shape[0]):
                    img_name = join(os.path.join(img_path, 'fids'), f"{img_idx}_{flow.use_ode_sampler}.{sample_config.file_ext}")
                    save_image(samples_t_end[img_idx:img_idx+1].clone().mul_(0.5).add_(0.5), img_name, nrow=1)
            
                # get all inception features for the second folder
                fdir2 = os.path.join(img_path, 'fids')
                fbname2 = os.path.basename(fdir2)
                np_feats2 = fid.get_folder_features(fdir2, feat_model, num_workers=num_workers,
                                                batch_size=batch_size, device=device_type, mode=mode,
                                                description=f"FID {fbname2} : ", verbose=verbose,
                                                custom_image_tranform=custom_image_transform,
                                                custom_fn_resize=custom_fn_resize)
                mu2 = np.mean(np_feats2, axis=0)
                sigma2 = np.cov(np_feats2, rowvar=False)
                # # save the data statistics
                # np.savez("data_stat.npz", mu2=mu2, sigma2=sigma2)
                score = fid.frechet_distance(mu1, sigma1, mu2, sigma2)
                # compute_fid
                logger.info(f"[FID] --> step: {global_step:08d}; fid: {score:0.6f}")
    
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    sys.exit(main())
