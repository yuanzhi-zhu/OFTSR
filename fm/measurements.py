'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from torch.nn import functional as F
from torchvision import torch
# from DPS.motionblur.motionblur import Kernel
# from DPS.util.img_utils import Blurkernel
import numpy as np


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device, **kwargs):
        self.device = device
    
    def forward(self, data, **kwargs):
        return data

    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data

    def project(self, data, **kwargs):
        return data

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

@register_operator(name='sr_avp')
class SuperResolutionAVPOperator(LinearOperator):
    def __init__(self, scale_factor, device, **kwargs):
        self.device = device
        self.up_sample =  lambda z: MeanUpsample(z,scale_factor)
        self.scale_factor = scale_factor

    def forward(self, data, **kwargs):
        h, w = data.shape[2], data.shape[3]
        down_sample = torch.nn.AdaptiveAvgPool2d((h//self.scale_factor, w//self.scale_factor)).to(self.device)
        return down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)


@register_operator(name='sr_interp')
class SuperResolutionInterpOperator(LinearOperator):
    def __init__(self, scale_factor, mode, device, **kwargs):
            self.device = device
            self.scale_factor = scale_factor
            self.mode = mode        # 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        
    def up_sample(self, z):
        return F.interpolate(z, scale_factor=self.scale_factor, mode=self.mode, antialias=True)
    
    def down_sample(self, data):
        return F.interpolate(data, scale_factor=1/self.scale_factor, mode=self.mode, antialias=True)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 


# =============
# Noise classes
# =============

__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma, **kwargs):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma

@register_noise(name='gaussian_VP')
class GaussianNoise(Noise):
    def __init__(self, sigma, **kwargs):
        self.sigma = sigma
    
    def forward(self, data):
        return np.sqrt(1 - self.sigma**2) * data + self.sigma * torch.randn_like(data, device=data.device)
