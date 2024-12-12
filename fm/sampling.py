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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""

import torch
import numpy as np
from fm.utils import from_flattened_numpy, to_flattened_numpy
from scipy import integrate
import logging
from typing import Any, Iterable, Tuple, Union

@torch.no_grad()
def get_flow_sampler(flow, use_ode_sampler=None, clip_denoised=False, device='cuda'):
    """
    Get rectified flow sampler

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    if use_ode_sampler is None:
        use_ode_sampler = flow.use_ode_sampler
    noise_var = flow.cfg.sample.noise_var
    rho = flow.cfg.sample.rho
    rk2_r = flow.cfg.sample.rk2_r

    @torch.no_grad()
    def one_step_sampler(model, z, **kwargs: Any,):
        """one_step_sampler.

        Args:
        model: A velocity model.
        z: If present, generate samples from latent code `z`.
        Returns:
        samples, number of function evaluations.
        """
        # Initial sample
        x, shape = z, z.shape
        
        ### one step
        eps = flow.eps # default: 1e-3
        t = torch.ones(shape[0], device=device) * eps
        with torch.autocast(device_type="cuda", enabled=flow.cfg.train.use_amp):
            pred = flow.model_forward_wrapper(model, x, t, **kwargs) ### Copy from models/utils.py 
        x = x + pred * (flow.T - eps)
        if clip_denoised:
            x = torch.clamp(x, -1, 1.)
        nfe = 1
        return x, nfe
        
    @torch.no_grad()
    def euler_sampler(model, z, return_xh=False, return_x1h=False, reverse=False, progress=False, **kwargs: Any,):
        """The probability flow ODE sampler with simple Euler discretization.

        Args:
        model: A velocity model.
        z: If present, generate samples from latent code `z`.
        noise_var: noise variance for turning ODE to SDE.
        Returns:
        samples, number of function evaluations.
        """
        # Initial sample
        x, shape = z, z.shape
        
        ### Uniform
        dt = 1./flow.sample_N
        eps = flow.eps # default: 1e-3
        x_h = []
        x_1_h = []
        t_h = []

        indices = range(flow.sample_N)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        if reverse:
            # For reverse, we need to modify the time stepping
            for i in indices:
                x_h.append(x.cpu())
                num_t = flow.T - (i / flow.sample_N * (flow.T - eps))
                t = torch.ones(shape[0], device=device) * num_t
                t_h.append(t.cpu())
                with torch.autocast(device_type="cuda", enabled=flow.cfg.train.use_amp):
                    pred = flow.model_forward_wrapper(model, x, t, **kwargs)
                x = x - pred * dt  # Note the negative sign here for reverse
        else:
            for i in indices:
                x_h.append(x.cpu())
                num_t = i / flow.sample_N * (flow.T - eps) + eps
                t = torch.ones(shape[0], device=device) * num_t
                t_h.append(t.cpu())
                with torch.autocast(device_type="cuda", enabled=flow.cfg.train.use_amp):
                    pred = flow.model_forward_wrapper(model, x, t, **kwargs)
                if noise_var > 0.:
                    # first derive the equstion for x_t = alpha_t x_0 + sigma_t x_1
                    # then swap t and (1-t) and change the sign of dt and pred
                    f_t   = -(1 / num_t) * x                    # f_t = d(log{\alpha_t})/dt * x
                    g_t_2 = 2 * (1-num_t) / num_t               # g_t^2 = d(\sigma_t^2)/dt - 2 * d(log{\alpha_t})/dt * \sigma_t^2
                    eps_t = np.sqrt(g_t_2) * noise_var
                    score = 2 * (f_t - (-pred)) / g_t_2         # v = f_t - 1/2 * g_t^2 * score
                    ## F-P equation: d{x} = (f_t - 1/2(g_t^2 + eps_t^2) * score) dt + eps_t d{w} = v dt - 1/2 eps_t^2 score dt + eps_t d{w}
                    x = x + pred * dt - 0.5 * (eps_t**2) * score * (-dt) + eps_t * np.sqrt(dt) * torch.randn_like(x)
                    ## similar to https://github.com/gnobitab/RectifiedFlow/blob/main/ImageGeneration/sampling.py#L97-L100
                else:
                    x = x + pred * dt
                x_1 = x + (1-num_t) * pred
                x_1_h.append(x_1.cpu())
        x_h.append(x.cpu())
        t_h.append(t.cpu())
        t_h = [t_[:1] for t_ in t_h]
        t_h = torch.cat(t_h, dim=0)
        t_h = np.array(t_h)
        if clip_denoised:
            x = torch.clamp(x, -1, 1.)
        nfe = flow.sample_N
        if return_xh:
            # to be consistent with rk45_sampler
            return x, nfe, (x_h, t_h)
        elif return_x1h:
            return x, nfe, (x_1_h, t_h[:-1])
        else:
            return x, nfe
        
    @torch.no_grad()
    def rk2_sampler(model, z, return_xh=False, reverse=False, progress=False, **kwargs: Any,):
        """The probability flow Generic second-order rk ODE sampler with simple Heun discretization.
        when r == 1, it is Heun's method.
        when r == 0.5, it is midpoint method.
        when r == 2/3, it is Ralston's method.
        Args:
        model: A velocity model.
        z: If present, generate samples from latent code `z`.
        Returns:
        samples, number of function evaluations.
        """
        assert not reverse, 'Not Implemented!'
        # Initial sample
        x, shape = z, z.shape
        
        ### Uniform
        # dt = 1./flow.sample_N
        eps = flow.eps # default: 1e-3
        x_h = []
        
        indices = torch.arange(flow.sample_N+1, device=device)
        timesteps = eps**rho + indices / max(flow.sample_N, 1) * (
            flow.T**rho - eps**rho
        )
        timesteps = timesteps**(1/rho)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            x_h.append(x.cpu())
            num_t = timesteps[i]
            num_t_next = timesteps[i+1] if i+1 < flow.sample_N else flow.T
            dt = num_t_next - num_t
            t = torch.ones(shape[0], device=device) * num_t
            with torch.autocast(device_type="cuda", enabled=flow.cfg.train.use_amp):
                pred = flow.model_forward_wrapper(model, x, t, **kwargs) ### Copy from models/utils.py 

            if num_t_next == flow.T:
                # Euler method
                x = x + pred * dt
            else:
                # RK2 method
                x_2 = x + rk2_r * pred * dt
                t_2 = torch.ones(shape[0], device=device) * (rk2_r * num_t_next + (1-rk2_r) * num_t)
                with torch.autocast(device_type="cuda", enabled=flow.cfg.train.use_amp):
                    pred_2 = flow.model_forward_wrapper(model, x_2, t_2, **kwargs)
                # pred_prime = ((1 - 1/(2*r)) * pred + 1/(2*r) * pred_2)
                pred_prime = (pred + 1/(2*rk2_r) * (pred_2 - pred))
                x = x + pred_prime * dt
        x_h.append(x.cpu())
        if clip_denoised:
            x = torch.clamp(x, -1, 1.)
        nfe = flow.sample_N * 2 - 1
        if return_xh:
            return x, nfe, x_h
        else:
            return x, nfe
        
    @torch.no_grad()
    def rk45_sampler(model, z, return_xh=False, reverse=False, **kwargs: Any,):
        """The probability flow ODE sampler with black-box ODE solver.
        by default adaptive rk45 method is used.

        Args:
        model: A velocity model.
        z: If present, generate samples from latent code `z`.
        Returns:
        samples, number of function evaluations.
        """
        rtol = atol = flow.ode_tol
        method = 'RK45'
        eps = flow.eps # default: 1e-3

        # Initial sample
        x, shape = z, z.shape

        @torch.no_grad()
        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t
            with torch.autocast(device_type="cuda", enabled=flow.cfg.train.use_amp):
                drift = flow.model_forward_wrapper(model, x, vec_t, **kwargs)
            # drift = model(x, vec_t*999)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        if reverse:
            solution = integrate.solve_ivp(ode_func, (flow.T, eps), to_flattened_numpy(x),
                                            rtol=rtol, atol=atol, method=method)
        else:
            solution = integrate.solve_ivp(ode_func, (eps, flow.T), to_flattened_numpy(x),
                                            rtol=rtol, atol=atol, method=method)
        nfe = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
        x_h = torch.tensor(solution.y.T)
        x_h = [xi.reshape(shape).type(torch.float32) for xi in x_h]
        if clip_denoised:
            x = torch.clamp(x, -1, 1.)
        if return_xh:
            return x, nfe, (x_h, solution.t)
        else:
            return x, nfe

    if use_ode_sampler == 'one_step':
        sample_N = 1
    elif use_ode_sampler == 'rk45':
        sample_N = "adaptive"
    else:
        sample_N = flow.sample_N
    logging.info(f'Type of Sampler: {use_ode_sampler}; sample_N: {sample_N}')
    if use_ode_sampler == 'one_step':
        return one_step_sampler
    elif use_ode_sampler == 'euler':
        return euler_sampler
    elif use_ode_sampler in ['rk2', 'midpoint', 'ralston', 'heun']:
        rk2_r = {'midpoint': 0.5, 'ralston': 2/3, 'heun': 1.}.get(use_ode_sampler, rk2_r)
        return rk2_sampler
    elif use_ode_sampler == 'rk45':
        return rk45_sampler
    else:
        assert False, 'Not Implemented!'
