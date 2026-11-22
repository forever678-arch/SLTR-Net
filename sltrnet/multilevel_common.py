"""Shared contracts for all eight post-fusion RF-layer decoder additions."""
import torch
from torch import nn
from torch.nn import functional as F


def norm(x):
    return F.layer_norm(x.movedim(1,-1),(x.shape[1],),eps=1e-5).movedim(-1,1)


def resize(x,size,mode='bilinear'):
    return F.interpolate(x,size=size,mode=mode,align_corners=False if mode=='bilinear' else None)


def masked(x,keep):
    if keep is None:return x
    if keep.ndim==3:keep=keep[:,None]
    support=resize(keep.float(),x.shape[-2:],'nearest').bool()
    return torch.where(support,x,torch.zeros((),device=x.device,dtype=x.dtype))


def stage_private_channels(stage):return {16:96,8:64,4:32}[stage]


def stage_index(stage):return {16:0,8:1,4:2}[stage]


def uniform_plan(g):return g.new_full((g.shape[0],6,4),.25,dtype=torch.float32)


def uniform_routes(a):return a.new_full((a.shape[0],4,*a.shape[-2:]),.25,dtype=torch.float32)


def zero_output(hidden=32,out=128):
    module=nn.Conv2d(hidden,out,1)
    nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
    return module


def masked_pool(x,keep,size):
    support=resize(keep.float(),x.shape[-2:],'nearest')
    mass=F.adaptive_avg_pool2d(support,size)
    mean=F.adaptive_avg_pool2d(masked(x,keep),size)/mass.clamp_min(1e-8).to(x.dtype)
    return torch.where(mass>0,mean,torch.zeros_like(mean)),mass
