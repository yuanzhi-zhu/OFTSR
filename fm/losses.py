# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
import logging


def get_optimizer(config, params):
    """Returns a flax optimizer object based on `config`."""
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps,
                            weight_decay=config.weight_decay)
    elif config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps,
                            weight_decay=config.weight_decay)
    elif config.optimizer == 'RAdam':
        optimizer = optim.RAdam(params, lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps,
                            weight_decay=config.weight_decay)
    elif config.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=config.lr, momentum=config.beta1, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError(
        f'Optimizer {config.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, params, step, lr=config.lr,
                    lr_decay=config.lr_decay,
                    warmup=config.warmup,
                    grad_clip=config.grad_clip,
                    amp_scaler=None):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        cur_lr = lr
        if lr_decay > 0:
            cur_lr = cur_lr / np.sqrt(max(step / lr_decay, 1))
        if warmup > 0:
            cur_lr = cur_lr * np.minimum(step / warmup, 1.0)
        for g in optimizer.param_groups:
            g['lr'] = cur_lr
        amp_scaler.unscale_(optimizer)
        if grad_clip > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            if torch.logical_or(total_norm.isnan(), total_norm.isinf()): 
                logging.info(f"Warning: total_norm is {total_norm} at step {step}")
        for param in params:
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        amp_scaler.step(optimizer)
        amp_scaler.update()

    return optimize_fn


def get_weightings(weight_schedule, snrs, t, sigma_data=0.5):
    """get weightings w(t) for different snr"""
    if weight_schedule == "snr":
        weightings = snrs
    if weight_schedule == "snr_inv":
        weightings = 1 / snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = torch.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = torch.ones_like(snrs)
    elif weight_schedule == "1mt":
        # D(x1, xt+(1-t)v(xt)) ==> D(x1, (1-t)x0+tx1+(1-t)v(xt))
        # ==> D((1-t)x1, (1-t)x0+(1-t)v(xt))
        weightings = (1 - t)**2 * torch.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


def get_flow_loss_fn(reduce_mean=True, loss_type='l2'):
    """Create a loss function for training with rectified flow.
    Args:
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    
    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(flow, predicted, target, noise=None, t=None):
        """Compute the loss function.
        Args:
        flow: A RectifiedFlow class object
        predicted: A mini-batch of predicted data.
        target: A mini-batch of target data.

        Returns:
        loss: A scalar that represents the average loss value across the mini-batch.
        """
        bs = target.shape[0]
        t = flow.t if t is None else t
        # get weight
        snrs = (t / (flow.T-t+flow.eps))**(2)
        weights = get_weightings(flow.cfg.train.weight_schedule, snrs, t)
        if 'lpips' in loss_type:
            if noise is None:    # reflow
                noise = flow.noise
                predicted_x = flow.xt + torch.einsum('b,bijk->bijk', (1 - t), predicted)
            else:    # distill
                predicted_x = noise + predicted
            target_x = noise + target
        # Compute losses
        if loss_type == 'l2':
            losses = torch.square(predicted - target)
        elif loss_type == 'l1':
            losses = torch.abs(predicted - target)
        elif 'pseudo_huber' in loss_type:
            c = float(loss_type.split('_')[-1]) if len(loss_type.split('_')) > 2 else 0.03
            losses = torch.sqrt(torch.square(predicted - target) + c**2) - c
        elif loss_type == 'lpips':
            losses = flow.lpips_forward_wrapper(predicted_x, target_x)
        elif loss_type == 'lpips+l2':
            lpips_losses = flow.lpips_forward_wrapper(predicted_x, target_x).view(bs, 1)
            l2_losses = torch.square(predicted - target).view(bs, -1).mean(dim=1, keepdim=True)
            losses = lpips_losses + l2_losses
        elif loss_type == 'lpips+l1':
            lpips_losses = flow.lpips_forward_wrapper(predicted_x, target_x).view(bs, 1)
            l1_losses = torch.abs(predicted - target).view(bs, -1).mean(dim=1, keepdim=True)
            losses = lpips_losses + l1_losses
        else:
            assert False, 'Not implemented'
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        losses *= weights
        loss = torch.mean(losses)
        return loss

    return loss_fn
