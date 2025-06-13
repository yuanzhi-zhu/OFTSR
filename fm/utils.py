import os
join = os.path.join
import torch
from torch import Tensor, nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image, make_grid
import yaml
import argparse
import subprocess
import shutil
import random
import logging
import cv2
from typing import Any, Union


def seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            self[key] = self._convert(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._convert(value))

    def _convert(self, value: Any) -> Any:
        if isinstance(value, dict):
            return EasyDict(value)
        elif isinstance(value, list):
            return [self._convert(item) for item in value]
        return value
    
    def __to_dict__(self):
        config_dict = {}
        for key, value in self.items():
            if isinstance(value, EasyDict):
                config_dict[key] = value.__to_dict__()
            else:
                config_dict[key] = value
        return config_dict

    def __has_attr__(self, name: str) -> bool:
        return name in self

def update_config(config, updates):
    """Recursively update a Config object."""
    for key, value in updates.items():
        keys = key.split('.')
        sub_config = config
        for k in keys[:-1]:
            sub_config = getattr(sub_config, k)
        setattr(sub_config, keys[-1], value)
    return config

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, help="Path to option YMAL file.")
    # Additional arguments for overriding config
    parser.add_argument("--overrides", nargs='+', help="Override config parameters. Use dot notation for nested fields, e.g., sample.sample_N=2")
    args = parser.parse_args()
    # Load the YAML file
    with open(args.opt, 'r') as file:
        config = yaml.safe_load(file)
    # Convert the dictionary to a EasyDict object
    config = EasyDict(config)
    if config.ir.degradation == 'realsr':
        config.degradation.sf = config.ir.scale_factor
        config.data.train.params.dir_paths = config.dataset.train_path
        config.data.val.params = config.data.train.params
        config.data.val.params.dir_paths = config.dataset.val_path
    # Process overrides
    if args.overrides:
        updates = {}
        for override in args.overrides:
            key, value = override.split('=')
            try:
                value = eval(value)  # Convert string to appropriate type
            except:
                pass
            updates[key] = value
        config = update_config(config, updates)
    config.world_size = torch.cuda.device_count()
    config.opt = args.opt

    return config


def restore_checkpoint(ckpt_dir, state, device, ema_only=False):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        if ema_only:
            state['ema'].load_state_dict(loaded_state['ema'])
        else:
            try:
                state['optimizer'].load_state_dict(loaded_state['optimizer']) if state['optimizer'] is not None else None
                state['model'].load_state_dict(loaded_state['model'], strict=False)
                state['ema'].load_state_dict(loaded_state['ema'])
                state['step'] = loaded_state['step']
                # state['model'] = state['model'].to(device)
                # state['ema'] = state['ema'].to(device)
            except:
                print("Error in loading training state. Load model and ema using ema weights.")
                state['ema'].load_state_dict(loaded_state['ema'])
                state['ema'].copy_to(state['model'].parameters())
        return state


def save_checkpoint(ckpt_dir, state, ema_only=False):
    if ema_only:
        saved_state = {
            'ema': state['ema'].state_dict(),
            'step': state['step']
        }
    else:
        saved_state = {
            'optimizer': state['optimizer'].state_dict(),
            'model': state['model'].state_dict(),
            'ema': state['ema'].state_dict(),
            'step': state['step']
        }
    torch.save(saved_state, ckpt_dir)


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def save_image_batch(batch, img_size, model_path, log_name="examples", normalize=False):
        sample_grid = make_grid(batch, nrow=int(np.ceil(np.sqrt(batch.shape[0]))), padding=img_size // 16)
        save_image(sample_grid, join(model_path, log_name), normalize=normalize)


def calc_psnr(batch1, batch2, max_pixel=2.0, eps=1e-10):
    mse = torch.mean((batch1 - batch2) ** 2, axis=(1, 2, 3))
    zeros = torch.zeros_like(mse)
    inf = torch.ones_like(mse) * float('inf')
    psnr_values = torch.where(mse == 0, inf, 20 * torch.log10(max_pixel / torch.sqrt(mse + eps)))
    psnr_values = torch.where(torch.isnan(psnr_values), zeros, psnr_values)
    mean_psnr = torch.mean(psnr_values)
    return mean_psnr.item()


def get_file_list():
    return [
        b.decode()
        for b in set(
            subprocess.check_output(
                'git ls-files -- ":!:load/*"', shell=True
            ).splitlines()
        )
        | set(  # hard code, TODO: use config to exclude folders or files
            subprocess.check_output(
                "git ls-files --others --exclude-standard", shell=True
            ).splitlines()
        )
    ]


def save_code_snapshot(model_path):
    os.makedirs(model_path, exist_ok=True)
    for f in get_file_list():
        if not os.path.exists(f) or os.path.isdir(f):
            continue
        os.makedirs(os.path.join(model_path, os.path.dirname(f)), exist_ok=True)
        shutil.copyfile(f, os.path.join(model_path, f))


if __name__ == "__main__":
    ## test by calculating the PSNR and LPIPS between two folder
    SR_path = 'SR_path'
    HR_path = 'HR_path'
    # get the file list
    SR_files = os.listdir(SR_path)
    HR_files = os.listdir(HR_path)
    SR_files.sort()
    HR_files.sort()
    # read data as tensor
    SR_data = torch.stack([torch.from_numpy(cv2.imread(os.path.join(SR_path, f)) / 255.0).permute(2, 0, 1) for f in SR_files])
    HR_data = torch.stack([torch.from_numpy(cv2.imread(os.path.join(HR_path, f)) / 255.0).permute(2, 0, 1) for f in HR_files])
    SR_data = SR_data.float() * 2 - 1
    HR_data = HR_data.float() * 2 - 1
    psnr = calc_psnr(SR_data, HR_data)
    
    import lpips
    device = torch.device('cuda')
    loss_fn_vgg = lpips.LPIPS(net='alex').to(device)
    with torch.no_grad():
        lpips_scores = 0
        lpips_bs = 8
        for i in range(int(np.ceil(HR_data.shape[0] / lpips_bs))):
            batch_samples = SR_data[i*lpips_bs:(i+1)*lpips_bs].to(device)
            batch_val_batch = HR_data[i*lpips_bs:(i+1)*lpips_bs].to(device)
            lpips_score = loss_fn_vgg(batch_samples, batch_val_batch)
            lpips_score = lpips_score.cpu().detach().numpy().mean()  # [-1,+1]
            lpips_scores += (lpips_score * batch_samples.shape[0])
        lpips_score = lpips_scores / HR_data.shape[0]
        
    print(f"PSNR: {psnr}, LPIPS: {lpips_score}")