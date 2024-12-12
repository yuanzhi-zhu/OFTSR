# -*- coding: utf-8 -*-
# Yuanzhi Zhu, 2023

import torch.nn as nn
import logging

__UNET__ = {}

def register_unet(name: str):
    def wrapper(cls):
        if __UNET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __UNET__[name] = cls
        return cls
    return wrapper


def create_model(config):
    config.attention_resolutions = "0" if not config.use_attention else config.attention_resolutions
    
    # Get the model creation function from the registry
    if __UNET__.get(config.model_arch, None) is None:
        raise NameError(f"Name {config.model_arch} is not defined.")
    model = __UNET__[config.model_arch](config)
    
    pytorch_total_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'total number of trainable parameters in model: {pytorch_total_grad_params}')
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'total number of parameters in model: {pytorch_total_params}')

    if config.world_size > 1:
        logging.info(f"using {config.world_size} GPUs!")

    return model



### model creater for diffuser UNet
@register_unet('diffuser')
def create_model_diffuser(config):
    from diffusers import UNet2DModel

    attention_resolutions = []
    for res in config.attention_resolutions.split(","):
        attention_resolutions.append(int(res))
        
    config.channel_mult = tuple(int(ch_mult) for ch_mult in config.channel_mult.split(","))
    ## get block_out_channels using model_channels and channel_mult
    block_out_channels = []
    for i in range(len(config.channel_mult)):
        block_out_channels.append(config.num_channels*config.channel_mult[i])
    block_out_channels = tuple(block_out_channels)
    ## get down_block_types and up_block_types using config.img_size, config.attn_resolutions and config.channel_mult
    down_block_types = []
    up_block_types = []
    for i in range(len(config.channel_mult)):
        res = config.img_size >> i
        if res in attention_resolutions:
            down_block_types.append("AttnDownBlock2D")
        else:
            down_block_types.append("DownBlock2D")
        if config.img_size // res in attention_resolutions:
            up_block_types.append("AttnUpBlock2D")
        else:
            up_block_types.append("UpBlock2D")
    down_block_types = tuple(down_block_types)
    up_block_types = tuple(up_block_types)
    ## create model
    return UNet2DModel(
                sample_size=config.img_size,
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                layers_per_block=config.num_res_blocks,
                block_out_channels=block_out_channels,
                down_block_types=down_block_types,
                up_block_types=up_block_types,
                norm_num_groups=min(32, config.num_channels),
                )
    

### create model with Openai guided diffusion UNet
### https://github.com/openai/guided-diffusion
@register_unet('guided_unet')
def create_model_guided_diff(config):
    from .unet import UNetModel

    learn_sigma = True
    use_checkpoint = False
    use_fp16 = False
    num_heads_upsample = -1
    num_head_channels = 64
    use_scale_shift_norm = True
    use_new_attention_order = False
    if config.channel_mult == "":
        if config.img_size == 512:
            config.channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif config.img_size == 256:
            config.channel_mult = (1, 1, 2, 2, 4, 4)
        elif config.img_size == 128:
            config.channel_mult = (1, 1, 2, 3, 4)
        elif config.img_size == 64:
            config.channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {config.img_size}")
    else:
        config.channel_mult = tuple(int(ch_mult) for ch_mult in config.channel_mult.split(","))

    attention_ds = []
    if config.use_attention:
        for res in config.attention_resolutions.split(","):
            attention_ds.append(config.img_size // int(res))

    return UNetModel(
        image_size=config.img_size,
        in_channels=config.in_channels,
        model_channels=config.num_channels,
        out_channels=(config.out_channels if not learn_sigma else 2*config.out_channels),
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=config.dropout,
        channel_mult=config.channel_mult,
        num_classes=config.num_classes,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=config.num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=config.resblock_updown,
        use_new_attention_order=use_new_attention_order,
        use_group_norm=config.use_group_norm,
        use_attention=config.use_attention,
        use_input_skip=config.use_input_skip,
        use_cond=config.use_cond,
    )
