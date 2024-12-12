# -*- coding: utf-8 -*-
# Yuanzhi Zhu, 2023

import copy
from typing import Any, Iterable, Tuple, Union
import numpy as np
import torch
from torch import Tensor, nn
import logging
from fm.measurements import get_noise, get_operator
from functools import partial
from fm.losses import get_flow_loss_fn
from fm.utils import calc_psnr
import lpips

PRECISION_MAP = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
}

class FM():
    def __init__(self, model=None, ema_model=None, cfg=None, device=None):
        self.cfg = cfg
        self.flow_config = cfg.fm_model
        self.config_ir = cfg.ir
        self.dataset_config = cfg.dataset
        self.network_config = cfg.network
        self.model = model
        self.ema_model = ema_model
        self.device = device
        
        self.use_ode_sampler = self.cfg.sample.use_ode_sampler
        self.psnr_batch_size = self.cfg.sample.psnr_batch_size
        self.sample_N = self.cfg.sample.sample_N
        self.ode_tol = self.cfg.sample.ode_tol
        self.eps = self.cfg.fm_model.eps
        self._T = self.cfg.fm_model.T
        self.amp_dtype = PRECISION_MAP.get(self.cfg.train.amp_dtype, torch.float16)     # default to fp16
        try:
            self.flow_t_schedule = int(self.cfg.fm_model.flow_t_schedule)
        except:
            self.flow_t_schedule = self.cfg.fm_model.flow_t_schedule
        ## get loss function
        self.loss_fn = get_flow_loss_fn(self.cfg.train.reduce_mean, self.cfg.train.loss_type)
        logging.info(f'sigma_y: {self.config_ir.sigma_y}')
        logging.info(f'sigma_pertubation: {self.config_ir.sigma_pertubation}')
        logging.info(f'ODE Tolerence: {self.ode_tol}')
        logging.info(f'use automatic mixed precision: {self.cfg.train.use_amp}; dtype: {self.amp_dtype}')
        self.amp_scaler = torch.amp.GradScaler(enabled=self.cfg.train.use_amp)
        ## get degradation operator
        self.operator = get_operator(name=self.config_ir.degradation, scale_factor=self.config_ir.scale_factor, mode=self.config_ir.mode, device=self.device)
        self.operator_val = get_operator(name=self.config_ir.degradation, scale_factor=self.config_ir.scale_factor, mode=self.config_ir.mode, device=self.device)
        self.noiser = get_noise(name='gaussian', sigma=self.config_ir.sigma_y)
        self.noiser_pertub = get_noise(name='gaussian_VP', sigma=self.config_ir.sigma_pertubation)
        ### lpips model for validation
        if 'lpips' in self.cfg.train.loss_type:
            self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)
        if self.config_ir.calc_LPIPS:
            self.loss_fn_vgg = lpips.LPIPS(net='alex').to(self.device)

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value

    def lpips_forward_wrapper(self, x, y, size=128):
        # https://github.com/richzhang/PerceptualSimilarity/issues/4#issuecomment-368344818
        # resize (B, C, H, W) to (B, C, 224, 224)
        if size > 0:
            # https://github.com/richzhang/PerceptualSimilarity/issues/104#issuecomment-1276672095
            x = nn.functional.interpolate(x, size=(size, size), mode='bilinear', antialias=True)
            y = nn.functional.interpolate(y, size=(size, size), mode='bilinear', antialias=True)
        return self.lpips_model(x, y)

    def model_forward_wrapper(
        self,
        model: nn.Module,
        x: Tensor,
        t: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Wrapper for the model call"""
        kwargs = {} if kwargs is None else kwargs
        label = kwargs['label'] if 'label' in kwargs else None
        augment_labels = kwargs['augment_labels'] if 'augment_labels' in kwargs else None
        x = torch.cat([x, self.cond], dim=1) if self.flow_config.use_cond else x
        model_output = model(x, t*999, label, augment_labels)
        model_output = model_output[0] if isinstance(model_output, tuple) else model_output
        return model_output[:, :3]
    
    def get_train_tuple_IR(self,
                            x: Tensor,
                            mask: Tensor = None,
                            **kwargs: Any,):
        """get sigma_t, x_t, sigma_{t+1}, x_{t+1}"""
        operator = self.operator if kwargs['mode'] == 'train' else self.operator_val
        ## Forward measurement model (Ax + n)
        y = operator.forward(x, mask=mask)
        yn_ = self.noiser(y)
        if self.config_ir.scale_factor > 1:
            yn_ = operator.transpose(yn_, mask=mask)
        self.cond = yn_.detach().clone()
        yn = self.noiser_pertub(yn_)     # noise pertubation
        self.x1 = x
        self.x0 = yn

    def get_interpolations(self,
                        data: Tensor,
                        noise: Tensor,
                        **kwargs: Any,):
        """get t, x_t based on the flow time schedule"""
        # sample timesteps
        if self.flow_t_schedule == 't0': ### distill for t = 0 (k=1)
            self.t = torch.zeros((data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif self.flow_t_schedule == 't1': ### reverse distill for t=1 (fast embedding)
            self.t = torch.ones((data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif self.flow_t_schedule == 't0t1': ### t = 0, 1, two ends of the trajectory
            self.t = torch.randint(0, 2, (data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif self.flow_t_schedule == 'uniform': ### train new rectified flow with reflow
            self.t = torch.rand((data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif type(self.flow_t_schedule) == float: ### train new rectified flow with fixed t
            self.t = torch.ones((data.shape[0],), device=data.device) * self.flow_t_schedule
        elif type(self.flow_t_schedule) == int: ### k > 1 
            self.t = torch.randint(0, self.flow_t_schedule, (data.shape[0],), device=data.device) * (self.T - self.eps) / self.flow_t_schedule + self.eps
        else:
            assert False, f'flow_t_schedule {self.flow_t_schedule} Not implemented'
        # linear interpolation between clean image and noise
        self.xt = torch.einsum('b,bijk->bijk', self.t, data) + torch.einsum('b,bijk->bijk', (1 - self.t), noise)

    def pred_tuple(self, **kwargs: Any,) -> Tuple[Tensor, Tensor]:
        # calculate next_x with model
        pred = self.model_forward_wrapper(
            self.model,
            self.xt,
            self.t,
            **kwargs,
        )
        # calculate target vector field
        target = self.x1 - self.x0
        return pred, target
    
    def train_step(self, batch, augment_pipe=None, mode='train', **kwargs: Any,):
        """Performs a training step"""
        ### get loss
        '''
        batch: Clean data.
        '''
        ## augment pipeline: edm
        ## --> https://github.com/NVlabs/edm/blob/main/training/augment.py
        batch, augment_labels = augment_pipe(batch) if augment_pipe is not None else (batch, None)
        ## get data pair (self.data, self.noise)
        self.get_train_tuple_IR(batch, mode=mode)
        ## get interpolation t, x_t
        self.get_interpolations(self.x1, self.x0)
        ## get prediction and target
        with torch.autocast(device_type="cuda", enabled=self.cfg.train.use_amp, dtype=self.amp_dtype):
            # kwargs['augment_labels'] = augment_labels
            predicted, target = self.pred_tuple(**kwargs)
            ## calculate loss
            loss = self.loss_fn(self, predicted, target)
        return loss

    # ------------------------------------ sampling ------------------------------------
    @torch.no_grad()
    def test_split_fn(self, x_0, cond, refield=32, min_size=256, modulo=1, sample_fn=None, t=0.001):
        '''
        model:
        x_0: augmented LR image
        cond: input Low-quality image --> measurement
        refield: effective receptive filed of the network, 32 is enough
        min_size: min_sizeXmin_size image, e.g., 256X256 image
        modulo: 1 if split
        '''
        h, w = x_0.size()[-2:]
        if h*w <= min_size**2:
            x_0 = torch.nn.ReplicationPad2d((0, int(np.ceil(w/modulo)*modulo-w), 0, int(np.ceil(h/modulo)*modulo-h)))(x_0)
            cond = torch.nn.ReplicationPad2d((0, int(np.ceil(w/modulo)*modulo-w), 0, int(np.ceil(h/modulo)*modulo-h)))(cond)
            E, nfe = self.image_restoration_wrapper(x_0, cond, sample_fn, t)
        else:
            top = slice(0, (h//2//refield+1)*refield)
            bottom = slice(h - (h//2//refield+1)*refield, h)
            left = slice(0, (w//2//refield+1)*refield)
            right = slice(w - (w//2//refield+1)*refield, w)
            x_0s = [x_0[..., top, left], x_0[..., top, right], x_0[..., bottom, left], x_0[..., bottom, right]]
            conds = [cond[..., top, left], cond[..., top, right], cond[..., bottom, left], cond[..., bottom, right]]

            if h * w <= 4*(min_size**2):
                Es = []
                nfe = 0
                for i in range(4):
                    E, n = self.image_restoration_wrapper(x_0s[i], conds[i], sample_fn, t)
                    Es.append(E)
                    nfe += n
                nfe = nfe / 4
            else:
                # Es = [self.test_split_fn(sample_fn, x_0s[i], conds[i], refield=refield, min_size=min_size, modulo=modulo, t=t) for i in range(4)]
                Es = []
                nfe = 0
                for i in range(4):
                    E, n = self.test_split_fn(x_0s[i], conds[i], refield=refield, min_size=min_size, modulo=modulo, sample_fn=sample_fn, t=t)
                    Es.append(E)
                    nfe += n
                nfe = nfe / 4

            b, c = Es[0].size()[:2]
            E = torch.zeros(b, c, h, w).type_as(x_0)

            E[..., :h//2, :w//2] = Es[0][..., :h//2, :w//2]
            E[..., :h//2, w//2:w] = Es[1][..., :h//2, (-w + w//2):]
            E[..., h//2:h, :w//2] = Es[2][..., (-h + h//2):, :w//2]
            E[..., h//2:h, w//2:w] = Es[3][..., (-h + h//2):, (-w + w//2):]
        return E, nfe


    @torch.no_grad()
    def image_restoration_(self, x_0, yn, sample_fn):
        samples = []
        nfe = []
        for i in range(int(np.ceil(x_0.shape[0] / self.psnr_batch_size))):
            batch_x0 = x_0[i*self.psnr_batch_size:(i+1)*self.psnr_batch_size]
            self.cond = yn.detach()[i*self.psnr_batch_size:(i+1)*self.psnr_batch_size]
            sample, n = sample_fn(self.model, z=batch_x0)
            samples.append(sample.cpu())
            nfe.append(n)
        samples = torch.cat(samples, dim=0)
        return samples, np.mean(nfe)

    @torch.no_grad()
    def image_restoration_t_(self, x_0, yn, t):
        samples = []
        for i in range(int(np.ceil(x_0.shape[0] / self.psnr_batch_size))):
            batch_x0 = x_0[i*self.psnr_batch_size:(i+1)*self.psnr_batch_size]
            vec_t = torch.ones(batch_x0.shape[0],).to(self.device) * t
            self.cond = yn.detach()[i*self.psnr_batch_size:(i+1)*self.psnr_batch_size]
            # sample, n = sample_fn(self.model, z=batch_x0)
            with torch.no_grad():
                v_pred = self.model_forward_wrapper(self.model, batch_x0, vec_t)
                sample = batch_x0 + (self.T - self.eps) * v_pred
            samples.append(sample.cpu())
        samples = torch.cat(samples, dim=0)
        return samples
    
    @torch.no_grad()
    def image_restoration_wrapper(self, x_0, yn, sample_fn=None, t=0.001):
        if self.cfg.sample.__has_attr__('use_one_step') and self.cfg.sample.use_one_step:
            samples = self.image_restoration_t_(x_0, yn, t=t)
            nfe = 1
        else:
            samples, nfe = self.image_restoration_(x_0, yn, sample_fn)
        return samples, nfe

    @torch.no_grad()
    def image_restoration(self, val_batch, sample_fn=None, t=0.001):
        # restore the images and calculate the PSNR and LPIPS
        y = self.operator_val.forward(val_batch.to(self.device))
        yn = self.noiser(y)
        y_LR = yn.clone()
        if self.config_ir.scale_factor > 1:
            yn = self.operator_val.transpose(yn)
        x_0 = self.noiser_pertub(yn)     # noise pertubation
        LR = (y_LR, yn, x_0)
        # samples, nfe = self.image_restoration_wrapper(x_0, yn, sample_fn, t)
        samples, nfe = self.test_split_fn(x_0, yn, sample_fn=sample_fn, t=t)
        psnr = calc_psnr(val_batch.cpu(), samples.cpu())
        if self.config_ir.calc_LPIPS:
            with torch.no_grad():
                lpips_scores = 0
                for i in range(int(np.ceil(val_batch.shape[0] / self.psnr_batch_size))):
                    batch_samples = samples[i*self.psnr_batch_size:(i+1)*self.psnr_batch_size].to(self.device)
                    batch_val_batch = val_batch[i*self.psnr_batch_size:(i+1)*self.psnr_batch_size].to(self.device)
                    lpips_score = self.loss_fn_vgg(batch_samples, batch_val_batch)
                    lpips_score = lpips_score.cpu().detach().numpy().mean()  # [-1,+1]
                    lpips_scores += (lpips_score * batch_samples.shape[0])
                lpips_score = lpips_scores / val_batch.shape[0]
        else:
            lpips_score = 0
        return psnr, lpips_score, samples, LR, np.mean(nfe)