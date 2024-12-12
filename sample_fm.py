# -*- coding: utf-8 -*-
# Yuanzhi Zhu, 2023

import os
import sys
join = os.path.join
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from fm.utils import (parse_args_and_config,
                      seed_everywhere,
                      restore_checkpoint)
from fm.image_datasets import ImageDataset
from fm import sampling as sampling
from models import create_model
from models.ema import ExponentialMovingAverage
from fm import FM
from datetime import datetime
import logging
from matplotlib import pyplot as plt

def save_batch_LR_SR(val_batch, samples, img_name, vis_num):
    sample_RLH = torch.cat([val_batch[:vis_num].cpu(), samples[:vis_num].cpu()], dim=3)
    sample_RLH = (sample_RLH + 1) / 2
    save_image(sample_RLH, img_name, nrow=1)

def main():
    ##################         prepare config        #####################
    config = parse_args_and_config()
    # Access the parameters
    dataset_config = config.dataset
    flow_config = config.fm_model
    network_config = config.network
    sample_config = config.sample
    dataset_config.in_channels = network_config.in_channels
    network_config.img_size = dataset_config.img_size
    network_config.num_classes = dataset_config.num_classes
    # network_config.in_channels = 2 * network_config.in_channels if flow_config.use_cond else network_config.in_channels
    network_config.world_size = torch.cuda.device_count()
    network_config.use_cond = flow_config.use_cond
    
    ##################          working paths           #####################
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    if 'checkpoints/' in sample_config.pre_train_model:
        work_dir = sample_config.pre_train_model.split('checkpoints')[0]
    else:
        work_dir = 'results'
        os.makedirs(work_dir, exist_ok=True)
    model_path = join(work_dir, f'eval_samples/{run_id}')
    os.makedirs(model_path, exist_ok=True)
    # set random seed everywhere
    seed_everywhere(config.seed)

    ### basic info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    ##################         setup loggers         #####################
    gfile_stream = open(f'{work_dir}/eval.log', 'a+')
    handler = logging.StreamHandler(gfile_stream)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(filename)s - %(asctime)s - %(levelname)s --> %(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    logger.setLevel('INFO')

    ##################       create & load model     #####################
    unet = create_model(network_config)
    unet.change_input_conv()
    unet = unet.to(device)
    ema = ExponentialMovingAverage(unet.parameters(), decay=network_config.ema_rate)
    state = dict(optimizer=None, model=unet, ema=ema, step=0)
    
    assert sample_config.pre_train_model
    state = restore_checkpoint(sample_config.pre_train_model, state, device, ema_only=True)
    logger.info(f'loading model from {sample_config.pre_train_model}')
    
    ##################         create flow            #####################
    flow = FM(model=unet, ema_model=ema, cfg=config, device=device)
    sampling_fn = sampling.get_flow_sampler(flow, device=device, use_ode_sampler=sample_config.use_ode_sampler)

    ##################         load ema weights       #####################
    ema.store(unet.parameters())
    unet.eval()
    ema.copy_to(unet.parameters())
    
    ##################          do evaluation         #####################
    if (not config.fm_model.use_cond) and config.ir.sigma_pertubation == 1.:
        ### image generation
        batch_shape = (sample_config.psnr_batch_size, network_config.in_channels, dataset_config.img_size, dataset_config.img_size)
        flow.cond = torch.zeros(batch_shape).to(device)
        for i in tqdm(range(sample_config.num_sample // sample_config.psnr_batch_size + 1)):
            x0 = torch.randn(batch_shape).to(device)
            with torch.no_grad():
                sample, n = sampling_fn(unet, z=x0)
            for img_idx in range(sample.shape[0]):
                if i*sample_config.psnr_batch_size + img_idx >= sample_config.num_sample:
                    break
                img_name = join(model_path, f"sample_{i*sample_config.psnr_batch_size+img_idx}_seed{config.seed}.{sample_config.file_ext}")
                save_image(sample[img_idx:img_idx+1]/2+0.5, img_name, nrow=1)
            logger.info(f"sample batch {i} --> nfe: {n}")
            # sample_grid = make_grid(sample/2+0.5, nrow=8, pad_value=1)
            # # save_image(sample/2+0.5, join(model_path, f"sample_{i}.{sample_config.file_ext}"), nrow=8)
            # save_image(sample_grid, join(model_path, f"sample_{i}.{sample_config.file_ext}"))
    else:
        ### image restoration
        # create folders
        os.makedirs(model_path + 'LR', exist_ok=True)
        os.makedirs(model_path + 'LR_yn', exist_ok=True)
        os.makedirs(model_path + 'LR_input_perturb', exist_ok=True)
        os.makedirs(model_path + 'HR', exist_ok=True)
        # create dataloaders
        img_dataset = ImageDataset(dataset_config, phase='val')
        data_loader = torch.utils.data.DataLoader(img_dataset,
                                                    batch_size=sample_config.psnr_batch_size,
                                                    shuffle=False,
                                                    num_workers=dataset_config.num_workers,
                                                    pin_memory= True)
        logger.info(f'evaluate from: {dataset_config.val_path}; length of img_dataset: {len(img_dataset)}')
        all_psnr = 0
        all_lpips = 0
        num_samples = 0
        nfe = 0
        for i, (val_batch, label_dic) in tqdm(enumerate(data_loader), total=len(data_loader)):
            num_samples += val_batch.shape[0]
            # if sample_config has use_one_step attribute, use one_step_t to control the step size
            t_in = sample_config.one_step_t if sample_config.__has_attr__('use_one_step') and sample_config.use_one_step else 0.001
            psnr, lpips_score, samples, LR, n = flow.image_restoration(val_batch, sampling_fn, t_in)
            y_LR, yn, x_0 = LR
            all_psnr += psnr * val_batch.shape[0]
            all_lpips += lpips_score * val_batch.shape[0]
            nfe += n * val_batch.shape[0]
            for img_idx in range(val_batch.shape[0]):
                img_name = join(model_path, f"{label_dic['img_name'][img_idx]}_{flow.use_ode_sampler}_seed{config.seed}.{sample_config.file_ext}")
                save_image(samples[img_idx:img_idx+1]/2+0.5, img_name, nrow=1)
                img_name_LR = join(model_path + 'LR', f"{label_dic['img_name'][img_idx]}_{flow.use_ode_sampler}.{sample_config.file_ext}")
                save_image(y_LR[img_idx]/2+0.5, img_name_LR.replace(f'_{flow.use_ode_sampler}','LR'), nrow=1)
                img_name_LR_yn = join(model_path + 'LR_yn', f"{label_dic['img_name'][img_idx]}_{flow.use_ode_sampler}.{sample_config.file_ext}")
                save_image(yn[img_idx]/2+0.5, img_name_LR_yn.replace(f'_{flow.use_ode_sampler}','LR_yn'), nrow=1)
                img_name_LR_p = join(model_path + 'LR_input_perturb', f"{label_dic['img_name'][img_idx]}_{flow.use_ode_sampler}.{sample_config.file_ext}")
                save_image(x_0[img_idx]/2+0.5, img_name_LR_p.replace(f'_{flow.use_ode_sampler}','LR_perturb'), nrow=1)
                img_name_Hr = join(model_path + 'HR', f"{label_dic['img_name'][img_idx]}_{flow.use_ode_sampler}.{sample_config.file_ext}")
                save_image(val_batch[img_idx:img_idx+1]/2+0.5, img_name_Hr.replace(f'_{flow.use_ode_sampler}','HR'), nrow=1)
            logger.info(f"batch {i} --> psnr: {psnr}; lpips: {lpips_score}; nfe: {n}; ave PSNR {all_psnr / num_samples}; ave lpips {all_lpips / num_samples}")
        all_psnr /= num_samples
        all_lpips /= num_samples
        nfe /= num_samples
        logger.info(f"[EVAL] --> steps: {state['step']}; psnr: {all_psnr}; lpips: {all_lpips}; nfe: {nfe}")
    
    # # compute_fid
    # from fid import calculate_fid
    # score = calculate_fid(model_path, dataset_config.val_path)
    # logger.info(f"[FID] --> fid: {score:0.6f}")
    logger.info(f"evaluation done! saved to {model_path}\n")

    # ## calculate straightness
    # N = 100
    # flow.sample_N = N
    # dt = 1. / N
    # sampling_fn = sampling.get_flow_sampler(flow, device=device, use_ode_sampler='euler')
    # val_batch_all = []
    # for i, (val_batch, label_dic) in tqdm(enumerate(data_loader), total=len(data_loader)):
    #     val_batch_all.append(val_batch)
    # val_batch = torch.cat(val_batch_all, dim=0)
    # # restore the images and calculate the PSNR and LPIPS
    # y = flow.operator_val.forward(val_batch.to(flow.device))
    # yn = flow.noiser(y)
    # if flow.config_ir.scale_factor > 1:
    #     yn = flow.operator_val.transpose(yn)
    # x_0 = flow.noiser_pertub(yn)
    # batch_x0 = x_0
    # flow.cond = yn.detach()
    # sample, n, (x_h, t_h) = sampling_fn(flow.model, z=batch_x0, return_xh=True, progress=True)
    # v_final = (sample - x_0).cpu() # [-1, 1]
    # straightness = []
    # for i in range(N):
    #     v_curr = (x_h[i+1] - x_h[i]) / dt
    #     # straight = torch.square(v_curr - v_final).view(v_curr.shape[0], -1).sum(dim=1)
    #     diff = (v_curr - v_final).view(v_curr.shape[0], -1)
    #     straight = torch.norm(diff, p='fro', dim=(1), keepdim=False)
    #     straightness.append(straight.mean() * dt)
    # straightness = torch.stack(straightness)
    # final_straightness = straightness.sum()
    # logger.info(f"straightness: {final_straightness}")
    # save_batch_LR_SR(val_batch[:4], sample[:4], join(model_path, f"straightness.{sample_config.file_ext}"), 1)

    # ## Plot pixel trajectories
    # seed_everywhere(config.seed)
    # num_pixel = 10
    # N = 1000
    # flow.sample_N = N
    # sampling_fn = sampling.get_flow_sampler(flow, device=device, use_ode_sampler='euler')

    # for i, (val_batch, label_dic) in tqdm(enumerate(data_loader), total=len(data_loader)):
    #     break
    # y = flow.operator_val.forward(val_batch.to(flow.device))
    # yn = flow.noiser(y)
    # if flow.config_ir.scale_factor > 1:
    #     yn = flow.operator_val.transpose(yn)
    # x_0 = flow.noiser_pertub(yn)
    # batch_x0 = x_0
    # flow.cond = yn.detach()[:1]
    # with torch.no_grad():
    #     x, nfe, (x_h, t_h) = sampling_fn(flow.model, z=batch_x0[:1], return_xh=True, progress=True)
    # print(t_h)
    # # Randomly sample n pixel positions
    # h_indices = torch.randint(0, x.shape[2], (num_pixel,))
    # w_indices = torch.randint(0, x.shape[3], (num_pixel,))
    # print(x_0[0, 0, h_indices, w_indices])
    # pixels_h = [x[0, 0, h_indices, w_indices].detach().cpu() for x in x_h]

    # plt.figure(figsize=(10, 5))
    # # For each position in the tensor, plot a curve
    # for position in range(pixels_h[0].shape[0]):
    #     plt.plot(t_h, [tensor[position] for tensor in pixels_h], label=f'pixel {position}')

    # plt.title("pixel trajectories")
    # plt.xlabel("time")
    # plt.ylabel("pixel value")
    # # plt.legend(loc="best")
    # plt.grid(True)
    # plt.savefig(os.path.join(model_path, f'euler_pixel_traj_{nfe}.png'), dpi=600, bbox_inches='tight')  # dpi determines resolution, bbox_inches ensures the entire plot is saved
    # plt.show()
    # # Close the current figure
    # plt.close()
    
if __name__ == "__main__":
    sys.exit(main())