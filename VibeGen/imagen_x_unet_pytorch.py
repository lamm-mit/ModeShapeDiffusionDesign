"""
Task:
1. create a unet for 1d data as a stand-alone function
2. consider replace UNet with a transformer net

Bo Ni, Aug 18, 2024
"""

# //////////////////////////////////////////////////////
# 0. load in packages
# //////////////////////////////////////////////////////

import math
from random import random
from beartype.typing import List, Union, Optional
from beartype import beartype
from tqdm.auto import tqdm
from functools import partial, wraps
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch import nn, einsum
from torch.cuda.amp import autocast
from torch.special import expm1
import torchvision.transforms as T

import kornia.augmentation as K

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# //////////////////////////////////////////////////////////////
# 1. local setup parameters: for debug purpose
# //////////////////////////////////////////////////////////////

# change the model setting to trigger those two

# 

# debug inside layers
IF_Debug_Depp_Layer = 0 # 0 # 1

# for those get deeper

Test_Debug_Level_1 = 1 # basic level, fundamental layers
# Default_Debug_Level == Test_Debug_Level_1: the debug mode is on



if IF_Debug_Depp_Layer==1:
    
    UNet_Init_Level = 0 # for initialization
    UNet_Forw_Level = 1 # for forward func
    
    Default_Debug_Level = 1
    
    # not a good practice, stop adding more
    Test_Debug_Level_2 = 0 # 1 # 1: show deeper layers during forword fun
    

else:
    UNet_Init_Level = 0 # for initialization
    UNet_Forw_Level = 0 # for forward func
    
    Default_Debug_Level = 0
    
    # # for those get deeper
    # Test_Debug_Level_1 = 0 # basic level, fundamental layers
    # # if Default_.. == Test_..., the debug mode is on
    
    Test_Debug_Level_2 = 0 # not show layers during forword fun
    
UNet_Init_Level = 1
UNet_Forw_Level = 1
    
# //////////////////////////////////////////////////////////////
# 2. supporting functions
# //////////////////////////////////////////////////////////////
# helper functions
def exists(val):
    return val is not None

# def identity(t, *args, **kwargs):
#     return t

# def divisible_by(numer, denom):
#     return (numer % denom) == 0

# def first(arr, d = None):
#     if len(arr) == 0:
#         return d
#     return arr[0]

# def maybe(fn):
#     @wraps(fn)
#     def inner(x):
#         if not exists(x):
#             return x
#         return fn(x)
#     return inner

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(val, length = None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output

def compact(input_dict):
    return {key: value for key, value in input_dict.items() if exists(value)}

def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)
        
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def resize_2d_image_to(
    image,
    target_image_size,
    clamp_range = None,
    mode = 'nearest'
):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    out = F.interpolate(image, target_image_size, mode = mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out

# //////////////////////////////////////////////////////////////
# 3. basic nets
# //////////////////////////////////////////////////////////////
# 
# =============================================================
# return a constant value
class Always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val
# 
# =============================================================
# return Identiy
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

# =============================================================    
# norms and residuals
# Channel root mean square normalization
class ChanRMSNorm_TwoD(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.scale * self.gamma
        # normalize x along the 2dn dim; self.gamma is trainable para
# ++
class ChanRMSNorm_OneD(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5 # here, dim refs to channel
        self.gamma = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.scale * self.gamma
        # normalize x along the 2dn dim; self.gamma is trainable para

# =============================================================
# layer normalization
# 
class LayerNorm(nn.Module):
    def __init__(
        self, 
        feats, 
        stable = False, 
        dim = -1
    ):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim = dim, keepdim = True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = dim, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = dim, keepdim = True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)
    # .rsqrt(): the reciprocal of the square-root of each of the elements of input
    
# ================================================================
# Apply multiple convolution operations with different width
# used as the inital involution pack: dimensional sensitive
class CrossEmbedLayer_TwoD(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv2d(
                    dim_in, 
                    dim_scale, 
                    kernel, 
                    stride = stride, 
                    padding = (kernel - stride) // 2
                )
            )
            
    def forward(
        self, 
        x,
        # ++
        Debug_Level = Default_Debug_Level,
    ):
        fmaps = tuple(
            map(lambda conv: conv(x), self.convs)
        )
        # ++
        if Debug_Level==Test_Debug_Level_1:
            print (f"    Inside CrossEmbedLayer:")
            for ii, this_map in enumerate(fmaps):
                print (f"    {ii} out.shape: {this_map.shape}")
        # 
        y = torch.cat(fmaps, dim = 1)
        # ++
        if Debug_Level==Test_Debug_Level_1:
            print (f"    out.shape: {y.shape}")
        return y
# ++
class CrossEmbedLayer_OneD(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes = None,
        dim_out = None,
        stride = 2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv1d(
                    dim_in,           # in_channels,
                    dim_scale,        # out_channels
                    kernel,           # kernel_size
                    stride = stride,  # stride
                    padding = (kernel - stride) // 2
                )
            )
            
    def forward(
        self, 
        x,
        # ++
        Debug_Level = Default_Debug_Level,
    ):
        fmaps = tuple(
            map(lambda conv: conv(x), self.convs)
        )
        # ++
        if Debug_Level==Test_Debug_Level_1:
            print (f"    Inside CrossEmbedLayer:")
            print (f"    input.shape: {x.shape}")
            for ii, this_map in enumerate(fmaps):
                print (f"    {ii} out.shape: {this_map.shape}")
        # 
        y = torch.cat(fmaps, dim = 1)
        # ++
        if Debug_Level==Test_Debug_Level_1:
            print (f"    out.shape: {y.shape}")
        return y

# 
# =====================================================
# not used
# 
# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
#         emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
#         return torch.cat((emb.sin(), emb.cos()), dim = -1)

# 
# =====================================================
# 
class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(
        self, dim,
        # ++
        debug_level=Default_Debug_Level,
    ):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))
        # ++
        self.debug_level = Default_Debug_Level

    def forward(
        self, 
        x,
        # ++
        Debug_Level = Default_Debug_Level,
    ):
        #++
        if self.debug_level==Test_Debug_Level_1:
            JJ=0
            print (f"    Inside LearnedSinusoidalPosEmb")
        
        x = rearrange(x, 'b -> b 1') # (b, 1)
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi # (b, d)
        #++
        if self.debug_level==Test_Debug_Level_1:
            JJ += 1
            print (f"      mid_{JJ} freqs.shape: {freqs.shape}")
        
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1) # (b, 2d)
        #++
        if self.debug_level==Test_Debug_Level_1:
            JJ += 1
            print (f"      mid_{JJ} fouriered.shape: {fouriered.shape}")
            
        fouriered = torch.cat((x, fouriered), dim = -1) # (b, 2d+1)
        #++
        if self.debug_level==Test_Debug_Level_1:
            print (f"    LearnedSinusoidalPosEmb ou.shape: {fouriered.shape}")
        return fouriered

# 
# ===========================================================
# used in ResNet
# attention pooling: dimension sensitive
class GlobalContext_TwoD(nn.Module):
    """ basically a superior form of squeeze-excitation that is attention-esque """

    def __init__(
        self,
        *,
        dim_in,
        dim_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(
            dim_in,  # in_channels
            1,       # out_channels
            1,       # kernel_size
        )
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x) # (b, 1, h, w)
        x, context = map(lambda t: rearrange(t, 'b n ... -> b n (...)'), (x, context))
        # context: (b, 1, h, w) -> (b, i=1, h*w)
        # x: (b, c, h, w) -> (b, c, h*w)
        out = einsum('b i n, b c n -> b c i', context.softmax(dim = -1), x)
        # out: (b, c, i=1)
        out = rearrange(out, '... -> ... 1')
        # out: (b, c, i=1, 1)
        return self.net(out)
# ++
class GlobalContext_OneD(nn.Module):
    """ basically a superior form of squeeze-excitation that is attention-esque """

    def __init__(
        self,
        *,
        dim_in,
        dim_out
    ):
        super().__init__()
        self.to_k = nn.Conv1d(
            dim_in,  # in_channels
            1,       # out_channels
            1,       # kernel_size
        )
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv1d(dim_in, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, dim_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x) # (b, 1, h)
        x, context = map(lambda t: rearrange(t, 'b n ... -> b n (...)'), (x, context))
        # context: (b, 1, h) -> (b, i=1, h)
        # x: (b, c, h) -> (b, c, h)
        out = einsum('b i n, b c n -> b c i', context.softmax(dim = -1), x)
        # out: (b, c, i=1)
        # --
        # out = rearrange(out, '... -> ... 1')
        
        return self.net(out)

# 
# ===========================================================
# ===========================================================
# for some attention layers: dimension insensitive
# 
# lauer builders:
# used on 
# PerceiverResampler, TransformerBlock, LinearAttentionTransformerBlock 
def FeedForward(dim, mult = 2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias = False)
    )
# 
# layer: put x through layers in and then sum them
# used in
# TwoD_Unet
class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)
# 
# function inside 
# PerceiverAttention, Attention, CrossAttention
def l2norm(t):
    return F.normalize(t, dim = -1)
# 
# used in PerceiverResampler
def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)
# 
# ===========================================================
# attention pooling: mainly operate on the embedding dim
class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.LayerNorm(dim)
        )

    def forward(self, x, latents, mask = None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # qk rmsnorm

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities and masking

        sim = einsum('... i d, ... j d  -> ... i j', q, k) * self.scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)
# 
# ===========================================================
# 
class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_latents_mean_pooled = 4, # number of latents derived from mean pooled representation of the sequence
        max_seq_len = 512,
        ff_mult = 4,
        # ++
        debug_level = Default_Debug_Level,
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.to_latents_from_mean_pooled_seq = None

        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange('b (n d) -> b n d', n = num_latents_mean_pooled)
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim = dim, 
                            dim_head = dim_head, 
                            heads = heads
                        ),
                        FeedForward(
                            dim = dim, 
                            mult = ff_mult
                        )
                    ]
                )
            )
        # ++
        self.debug_level = debug_level

    def forward(self, x, mask = None):
        
        n, device = x.shape[1], x.device
        # ++
        if self.debug_level==Test_Debug_Level_1:
            print (f"    Inside PerceiverResampler")
            JJ=0
            print (f"      x.shape: {x.shape}")
            print (f"      n as the seq_len: {n}")
            
        pos_emb = self.pos_emb(
            torch.arange(n, device = device)
        )
        # ++
        if self.debug_level==Test_Debug_Level_1:
            JJ+=1
            print (f"     mid_{JJ}: pos_emb.shape: {pos_emb.shape}")
        

        x_with_pos = x + pos_emb
        # ++
        if self.debug_level==Test_Debug_Level_1:
            JJ+=1
            print (f"      mid_{JJ}: x+pos_emb->x.shape: {x.shape}")

        latents = repeat(
            self.latents, 'n d -> b n d', b = x.shape[0]
        )
        # ++
        if self.debug_level==Test_Debug_Level_1:
            JJ+=1
            print (f"      mid_{JJ}: latents.shape: {latents.shape}")

        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(
                x, dim = 1, 
                mask = torch.ones(
                    x.shape[:2], 
                    device = x.device, 
                    dtype = torch.bool
                )
            )
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim = -2)
            # ++
            if self.debug_level==Test_Debug_Level_1:
                JJ+=1
                print (f"      mid_{JJ}: x->meanpooled_seq.shape: {meanpooled_seq.shape}")
                print (f"      mid_{JJ}: meanpooled_latents.shape: {meanpooled_latents.shape}")
                print (f"      mid_{JJ}: cat(lat.., mean..)->latents.shape: {latents.shape}")

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask = mask) + latents
            latents = ff(latents) + latents
            # ++
            if self.debug_level==Test_Debug_Level_1:
                JJ+=1
                print (f"      mid_{JJ}: attn+ff=>latents.shape: {latents.shape}")

        return latents
# 
# ===========================================================
# Used in UNet
# dimension sensitive
class PixelShuffleUpsample_TwoD(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)
    
def Downsample_TwoD(dim, dim_out = None):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )

def Upsample_TwoD(dim, dim_out = None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )
# 
# ===========================================================
# ++
def Downsample_OneD(dim, dim_out = None):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle
    #
    # scale factor is 2 here.
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b c (h s1) -> b (c s1) h', s1 = 2), # reduce dimension, add them into channels,
        # dim_in is amplified by dimension
        nn.Conv1d(dim * 2, dim_out, 1)
    )
# 
def Upsample_OneD(dim, dim_out = None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, dim_out, 3, padding = 1)
    )
# 
# ===========================================================
# dimension sensitive
# to replace nn.PixelShuffle for 1d data
class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width
        
        # expand the channel dimension
        # initially, x.shape (batch_size, short_channel, short_width)
        x = x.contiguous().view(
            [batch_size, self.upscale_factor, long_channel_len, short_width]
        ) 
        # (batch_size, upscale_factor, long_channel_len, short_width)
        # (batch_size, long_channel_len, short_width, upscale_factor)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x
    
class PixelShuffleUpsample_OneD(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        # --
        # conv = nn.Conv2d(dim, dim_out * 4, 1)
        # ++
        conv = nn.Conv1d(dim, dim_out * 2, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            PixelShuffle1D(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h = conv.weight.shape
        conv_weight = torch.empty(o // 2, i, h)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 2) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)
# ===========================================================
# 

ChanLayerNorm_TwoD = partial(LayerNorm, dim = -3)
ChanLayerNorm_OneD = partial(LayerNorm, dim = -2)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 8,
        dropout = 0.05,
        context_dim = None,
        **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm_OneD(dim)

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(dim, inner_dim, 1, bias = False),
            nn.Conv1d(
                inner_dim, inner_dim, 3, bias = False, 
                padding = 1, groups = inner_dim
            )
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(dim, inner_dim, 1, bias = False),
            nn.Conv1d(
                inner_dim, inner_dim, 3, bias = False, 
                padding = 1, groups = inner_dim
            )
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(dim, inner_dim, 1, bias = False),
            nn.Conv1d(
                inner_dim, inner_dim, 3, bias = False, 
                padding = 1, groups = inner_dim
            )
        )

        self.to_context = nn.Sequential(
            nn.LayerNorm(context_dim), 
            nn.Linear(context_dim, inner_dim * 2, bias = False)
        ) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm_OneD(dim)
        )

    def forward(
        self, fmap, 
        context = None,
        # ++
        Debug_Level = Default_Debug_Level,
    ):
        # ++
        # print (f"Debug_Level: {Debug_Level}")
        # print (f"Test_Debug_Level: {Test_Debug_Level_1}")
        if Debug_Level == Test_Debug_Level_1:
            print (f"Inside a LinearAttension.forward:...")
            print (f"fmap.shape: {fmap.shape}")
            
        # --
        # h, x, y = self.heads, *fmap.shape[-2:]
        # ++
        h, x = self.heads, fmap.shape[-1]
        if Debug_Level == Test_Debug_Level_1:
            print (f"self.heads->h.shape: {h}")
            print (f"x as seq_len: {x}")

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        # ++
        if Debug_Level == Test_Debug_Level_1:
            print (f".to_q->q.shape: {q.shape}") # (batch, (head channel), x)
            print (f".to_k->k.shape: {k.shape}")
            print (f".to_v->v.shape: {v.shape}")
        # -- 2d case   
        # q, k, v = map(
        #     lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v)
        # )
        # ++
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x -> (b h) x c', h = h), (q, k, v)
        )
        if Debug_Level == Test_Debug_Level_1:
            print (f"After reshape to put head and batch toghter:")
            print (f"q.shape: {q.shape}") # (batch, (head channel), x)
            print (f"k.shape: {k.shape}")
            print (f"v.shape: {v.shape}")

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            ck, cv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (ck, cv))
            k = torch.cat((k, ck), dim = -2)
            v = torch.cat((v, cv), dim = -2)
            # ++
            if Debug_Level == Test_Debug_Level_1:
                print (f"After adding context, k v.shape: {k.shape}, {v.shape}")

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        # --
        # out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)
        # ++ 
        out = rearrange(out, '(b h) x d -> b (h d) x', h = h, x = x)
        if Debug_Level == Test_Debug_Level_1:
            print (f"out.shape: {out.shape}")

        out = self.nonlin(out)
        if Debug_Level == Test_Debug_Level_1:
            print (f".nonlin(out)->out.shape: {out.shape}")
            print (f".to_out(out)->out.shape: {self.to_out(out).shape}")
        return self.to_out(out)
    
def ChanFeedForward_OneD(dim, mult = 2):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm_OneD(dim),
        nn.Conv1d(dim, hidden_dim, 1, bias = False),
        nn.GELU(),
        ChanLayerNorm_OneD(hidden_dim),
        nn.Conv1d(hidden_dim, dim, 1, bias = False)
    )

# 
# ===========================================================
# ===========================================================
# RsenetBlock: dimension sensitive
# 
# this one includes,
# normalization, activation, then convolution
class Block_TwoD(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        norm = True
    ):
        super().__init__()
        self.norm = ChanRMSNorm_TwoD(dim) if norm else Identity()
        self.activation = nn.SiLU() # Sigmoid linear unit, element-wise
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)
# 
# ===========================================================
# 
class ResnetBlock_TwoD(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim = None,
        time_cond_dim = None,
        linear_attn = False,
        use_gca = False,
        squeeze_excite = False,
        # ++
        debug_level = Default_Debug_Level,
        # 
        **attn_kwargs
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            self.cross_attn = attn_klass(
                dim = dim_out,
                context_dim = cond_dim,
                **attn_kwargs
            )

        self.block1 = Block_TwoD(dim, dim_out)
        self.block2 = Block_TwoD(dim_out, dim_out)

        self.gca = GlobalContext_TwoD(
            dim_in = dim_out, dim_out = dim_out
        ) if use_gca else Always(1)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()
        # ++
        self.debug_level = debug_level

    def forward(
        self, 
        x, 
        time_emb = None, 
        cond = None
    ):
        # ++
        if self.debug_level == Test_Debug_Level_1:
            print (f"    In ResnetBlock:")

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1) 
            # list of tensors ((b, c/2, 1, 1), (b, c/2, 1, 1)): scale, shift
            # ++
            if self.debug_level == Test_Debug_Level_1:
                print (f"    time_emb.shape: {time_emb.shape} as (b,2*in_channle,1,1)" ) # (b, c, 1, 1)
                # print (f"scale_shift.shape: ")

        h = self.block1(x) # first N+A+convolution: Convo with fixed windows length
        # ++
        if self.debug_level == Test_Debug_Level_1:
            print (f"    ResnetBlock.block1(x)->h.shape: {h.shape}")

        if exists(self.cross_attn):
            assert exists(cond)
            h = rearrange(h, 'b c h w -> b h w c')
            h, ps = pack([h], 'b * c') # h.shape: (b, h*w, c)
            h = self.cross_attn(h, context = cond) + h # (b, h*w, c)
            h, = unpack(h, ps, 'b * c') # h.shape: (b, h, w, c)
            h = rearrange(h, 'b h w c -> b c h w')
            # ++
            if self.debug_level == Test_Debug_Level_1:
                print (f"    Cross attention is included via conditioning")

        h = self.block2(h, scale_shift = scale_shift)
        # ++
        if self.debug_level == Test_Debug_Level_1:
            print (f"    ResnetBlock.block2(h)->h.shape: {h.shape}")

        h = h * self.gca(h)
        # ++
        if self.debug_level == Test_Debug_Level_1:
            print (f"    h * ResnetBlock.gca(h)->h.shape: {h.shape}")

        return h + self.res_conv(x)
# 
# ===========================================================
# ++
# this one includes,
# normalization, activation, then convolution
class Block_OneD(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        norm = True
    ):
        super().__init__()
        self.norm = ChanRMSNorm_OneD(dim) if norm else Identity()
        self.activation = nn.SiLU() # Sigmoid linear unit, element-wise
        # --
        # self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)
        # ++
        self.project = nn.Conv1d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)
# 
# ===========================================================
# ++ for 1D
# 
class ResnetBlock_OneD(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim = None,
        time_cond_dim = None,
        linear_attn = False,
        use_gca = False,
        squeeze_excite = False,
        # ++
        debug_level = Default_Debug_Level,
        # 
        **attn_kwargs
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            self.cross_attn = attn_klass(
                dim = dim_out,
                context_dim = cond_dim,
                **attn_kwargs
            )

        self.block1 = Block_OneD(dim, dim_out)
        self.block2 = Block_OneD(dim_out, dim_out)

        self.gca = GlobalContext_OneD(
            dim_in = dim_out, dim_out = dim_out
        ) if use_gca else Always(1)
        # --
        # self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()
        # ++
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else Identity()
        # ++
        self.debug_level = debug_level

    def forward(
        self, 
        x, 
        time_emb = None, 
        cond = None
    ):
        # ++
        if self.debug_level == Test_Debug_Level_1:
            print (f"    In ResnetBlock:")

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            # --
            # time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # ++
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1) 
            # list of tensors ((b, c/2, 1), (b, c/2, 1)): scale, shift
            # ++
            if self.debug_level == Test_Debug_Level_1:
                print (f"        time_emb.shape: {time_emb.shape} as (b,2*in_channle,1)" ) # (b, c, 1, 1)
                # print (f"scale_shift.shape: ")

        h = self.block1(x) # first N+A+convolution: Convo with fixed windows length
        # ++
        if self.debug_level == Test_Debug_Level_1:
            print (f"        ResnetBlock.block1(x)->h.shape: {h.shape}")

        if exists(self.cross_attn):
            assert exists(cond)
            # --
            # h = rearrange(h, 'b c h w -> b h w c')
            # h, ps = pack([h], 'b * c') # h.shape: (b, h*w, c)
            # h = self.cross_attn(h, context = cond) + h # (b, h*w, c)
            # h, = unpack(h, ps, 'b * c') # h.shape: (b, h, w, c)
            # h = rearrange(h, 'b h w c -> b c h w')
            # ++
            h = rearrange(h, 'b c h -> b h c') # h.shape: (b, h, c)
            # h, ps = pack([h], 'b * c') # h.shape: (b, h*w, c), w=1
            h = self.cross_attn(h, context = cond) + h # (b, h*w, c), w=1
            # h, = unpack(h, ps, 'b * c') # h.shape: (b, h, w, c)
            # --
            # h = rearrange(h, 'b h w c -> b c h w')
            # ++
            h = rearrange(h, 'b h c -> b c h')
            # ++
            if self.debug_level == Test_Debug_Level_1:
                print (f"        Cross attention is included via conditioning")

        h = self.block2(h, scale_shift = scale_shift)
        # ++
        if self.debug_level == Test_Debug_Level_1:
            print (f"        ResnetBlock.block2(h)->h.shape: {h.shape}")

        h = h * self.gca(h)
        # ++
        if self.debug_level == Test_Debug_Level_1:
            print (f"        h * ResnetBlock.gca(h)->h.shape: {h.shape}")

        return h + self.res_conv(x)

# 
# ===========================================================
# ===========================================================
# Attention blocks
#
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        context_dim = None,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_context = nn.Sequential(
            nn.LayerNorm(context_dim), 
            nn.Linear(context_dim, dim_head * 2)
        ) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(
        self, x, 
        context = None, 
        mask = None, 
        attn_bias = None,
        # 
        Debug_Level = Default_Debug_Level,
    ):
        b, n, device = *x.shape[:2], x.device
        if Debug_Level == Test_Debug_Level_1:
            print (f"In Attention layer:")
            print (f"    x.shape: {x.shape}")

        x = self.norm(x) # (b, n, (h*d))

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        # q.dim: ()
        if Debug_Level == Test_Debug_Level_1:
            print (f"    q.shape: {q.shape}")
            print (f"    k.shape: {k.shape}")
            print (f"    v.shape: {v.shape}")

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(
            lambda t: repeat(t, 'd -> b 1 d', b = b), self.null_kv.unbind(dim = -2)
        )
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)
        if Debug_Level == Test_Debug_Level_1:
            print (f"    nk.shape: {nk.shape}")
            print (f"    nv.shape: {nv.shape}")
            print (f"    k.shape: {k.shape}")
            print (f"    v.shape: {v.shape}")

        # add text conditioning, if present

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        # qk rmsnorm

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale
        if Debug_Level == Test_Debug_Level_1:
            print (f"    sim.shape: {sim.shape}")

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)
        if Debug_Level == Test_Debug_Level_1:
            print (f"    attn.shape: {attn.shape}")

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)
        if Debug_Level == Test_Debug_Level_1:
            print (f"    out.shape: {out.shape}")

        out = rearrange(out, 'b h n d -> b n (h d)')
        if Debug_Level == Test_Debug_Level_1:
            print (f"    out.shape: {out.shape}")
            
        out = self.to_out(out)
        if Debug_Level == Test_Debug_Level_1:
            print (f"    out.shape: {out.shape}")
        
        return out
# 
# ===========================================================
# 
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        norm_context = False,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(
        self, x, 
        context, 
        mask = None,
        # 
        Debug_Level = Default_Debug_Level,
    ):
        
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v)
        )

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(
            lambda t: repeat(t, 'd -> b h 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2)
        )

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # cosine sim attention

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
# 
# ===========================================================
# 
class LinearCrossAttention(CrossAttention):
    # 
    def forward(
        self, x, 
        context, 
        mask = None
    ):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.heads), (q, k, v))

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> (b h) 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # masking

        max_neg_value = -torch.finfo(x.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b n -> b n 1')
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.)

        # linear attention

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = self.heads)
        return self.to_out(out)
# 
# ===========================================================
#
class TransformerBlock_TwoD(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        context_dim = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim = dim, heads = heads, 
                            dim_head = dim_head, 
                            context_dim = context_dim
                        ),
                        FeedForward(
                            dim = dim, mult = ff_mult
                        )
                    ]
                )
            )

    def forward(self, x, context = None):
        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')

        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')
        return x
# ++
class TransformerBlock_OneD(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        context_dim = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim = dim, heads = heads, 
                            dim_head = dim_head, context_dim = context_dim
                        ),
                        FeedForward(
                            dim = dim, mult = ff_mult
                        )
                    ]
                )
            )

    def forward(self, x, context = None):
        # --
        # x = rearrange(x, 'b c h w -> b h w c')
        # x, ps = pack([x], 'b * c')
        # ++
        x = rearrange(x, 'b c h -> b h c')

        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        # --
        # x, = unpack(x, ps, 'b * c')
        # x = rearrange(x, 'b h w c -> b c h w')
        # ++
        x = rearrange(x, 'b h c -> b c h')
        return x
# 
# ===========================================================
#
class LinearAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        context_dim = None,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        LinearAttention(
                            dim = dim, 
                            heads = heads, 
                            dim_head = dim_head, 
                            context_dim = context_dim,
                        ),
                        ChanFeedForward(
                            dim = dim, 
                            mult = ff_mult
                        )
                    ]
                )
            )

    def forward(self, x, context = None):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x
# 
# ===========================================================
# 
class UpsampleCombiner_TwoD(nn.Module):
    def __init__(
        self,
        dim,
        *,
        enabled = False,
        dim_ins = tuple(),
        dim_outs = tuple()
    ):
        super().__init__()
        dim_outs = cast_tuple(dim_outs, len(dim_ins))
        assert len(dim_ins) == len(dim_outs)

        self.enabled = enabled

        if not self.enabled:
            self.dim_out = dim
            return

        self.fmap_convs = nn.ModuleList(
            [
                Block_TwoD(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)
            ]
        )
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def forward(self, x, fmaps = None):
        target_size = x.shape[-1]

        fmaps = default(fmaps, tuple())

        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x

        fmaps = [resize_2d_image_to(fmap, target_size) for fmap in fmaps]
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        return torch.cat((x, *outs), dim = 1)
    
# ++
# just passed from the twoD case: need to check for OneD
def resize_image_to(
    image,
    target_image_size,
    clamp_range = None,
    mode = 'nearest'
):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    out = F.interpolate(image, target_image_size, mode = mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out


class UpsampleCombiner_OneD(nn.Module):
    def __init__(
        self,
        dim,
        *,
        enabled = False,
        dim_ins = tuple(),
        dim_outs = tuple(),
        # ++
        debug_level = Default_Debug_Level,
    ):
        super().__init__()
        dim_outs = cast_tuple(dim_outs, len(dim_ins))
        assert len(dim_ins) == len(dim_outs)

        self.enabled = enabled

        if not self.enabled:
            self.dim_out = dim
            return

        self.fmap_convs = nn.ModuleList(
            [
                Block_OneD(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)
            ]
        )
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)
        # ++
        self.debug_level = debug_level

    def forward(
        self, x, 
        fmaps = None,
    ):
        
        target_size = x.shape[-1]
        # ++
        if self.debug_level == Test_Debug_Level_1:
            print (f"Inside UpsampleCombiner_OneD:")
            print (f"    target_size: {target_size}")

        fmaps = default(fmaps, tuple())
        # ++
        if self.debug_level == Test_Debug_Level_1:
            if len(fmaps) == 0:
                print (f"    fmaps: None")
                print (f"    nothing add to x")
            else:
                print (f"    list fmaps:")
                for fmap in fmaps:
                    print (f"    fmap.shape: {fmap.shape}")

        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x

        fmaps = [resize_image_to(fmap, target_size) for fmap in fmaps]
        # ++
        if self.debug_level == Test_Debug_Level_1:
            print (f"    After resize_image_to:")
            for fmap in fmaps:
                print (f"    fmap.shape: {fmap.shape}")
                
        #? resize_image_to need a 1d version, not not used at this moment
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        # ++
        if self.debug_level == Test_Debug_Level_1:
            print (f"    After .fmap_convs, get outs")
            print (f"    list out in outs:")
            for this_out in outs:
                print (f"    out.shape: {this_out.shape}")
            print (f"    next, do torch.cat((x, *outs), dim = 1) ")
            
        return torch.cat((x, *outs), dim = 1)

# 
# ===========================================================
# ===========================================================
# 
# 

# //////////////////////////////////////////////////////////////
# 4. key nets
# //////////////////////////////////////////////////////////////
# 
# ===========================================================
# ===========================================================
#
# null unet

class NullUnet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lowres_cond = False
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))

    def cast_model_parameters(self, *args, **kwargs):
        return self

    def forward(self, x, *args, **kwargs):
        return x

# place-holder: for future: t-dependent sequence

class Unet3D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lowres_cond = False
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))

    def cast_model_parameters(self, *args, **kwargs):
        return self

    def forward(self, x, *args, **kwargs):
        # x.shape: (batch, channel, frame, seq_len)
        assert x.dim()==4, "Wrong dim for input."
        return x

# from imagen_video
# TBU
def resize_video_to(
    video,
    target_image_size,
    target_frames = None,
    clamp_range = None,
    mode = 'nearest'
):
    orig_video_size = video.shape[-1]

    frames = video.shape[2]
    target_frames = default(target_frames, frames)

    target_shape = (target_frames, target_image_size, target_image_size)

    if tuple(video.shape[-3:]) == target_shape:
        return video

    out = F.interpolate(video, target_shape, mode = mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)
        
    return out

# from imagen_video
# TBU
def scale_video_time(
    video,
    downsample_scale = 1,
    mode = 'nearest'
):
    if downsample_scale == 1:
        return video

    image_size, frames = video.shape[-1], video.shape[-3]
    assert divisible_by(frames, downsample_scale), f'trying to temporally downsample a conditioning video frames of length {frames} by {downsample_scale}, however it is not neatly divisible'

    target_frames = frames // downsample_scale

    resized_video = resize_video_to(
        video,
        image_size,
        target_frames = target_frames,
        mode = mode
    )

    return resized_video

# ===========================================================
# ===========================================================
#
# try 1D case
#
class Unet_OneD(nn.Module):
    def __init__(
        self,
        *,
        # 1. in-out channels
        channels = 3,                  # channel of image-in
        channels_out = None,           # channel of image output
        
        # 2. on initial convolutions: CrossEmbedLayer
        # 2.1 input dim
        lowres_cond = False,           # for cascading diffusion - https://cascaded-diffusion.github.io/ low resolution image as condition
        self_cond = False,             # use self-condition or not
        cond_images_channels = 0,      # channels of conditioning image,> 0 to activate
        # 2.2 output dim
        init_dim = None,                    # inital emb dim. If not set, use dim
        # 2.3 layer structures
        init_cross_embed = True,       # use CrossEmbedLayer otherwise just Conv
        init_cross_embed_kernel_sizes = (3, 7, 15), # if CrossEmbedLayer is used
        init_conv_kernel_size = 7,                  # kernel size of initial conv, if not using cross embed
        
        # 3. prepare embedding
        # 3.1 on time_hiddens, time_cond, time_tokens
        cond_dim = None,               # embedding dim of condition
        dim,                           # emb_dim
        dim_mults=(1, 2, 4, 8),        # in ResNet, chang image H W
        learned_sinu_pos_emb_dim = 16, # pos emb dim
        num_time_tokens = 2,           # num of time tokens
        # 3.2 on lowres_cond
        # 3.3 on text conditioning
        text_embed_dim = 768,          # get_encoded_dim(DEFAULT_T5_NAME),
        # finer control over whether to condition on text encodings
        cond_on_text = True,
        # if activate Perceiver re-sampler
        attn_pool_text = True,         # activate Perceiver re-sampler if true
        attn_dim_head = 64,            # attation head dim
        attn_heads = 8,                # attation head number
        attn_pool_num_latents = 32,    # num_latents
        # for classifier free guidance
        max_text_len = 256,
        
        # ResNet parts
        num_resnet_blocks = 1,         # num of ResNet blocks
        layer_attns = True,
        layer_attns_depth = 1,
        layer_cross_attns = True,
        use_linear_attn = False,
        use_linear_cross_attn = False,
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        memory_efficient = False,
        use_global_context_attn = True,
        scale_skip_connection = True,
        pixel_shuffle_upsample = True,       # may address checkboard artifacts
        ff_mult = 2.,                        # feed forward in down/up-sample attension
        layer_mid_attns_depth = 1,           # middle layers in ResNet
        attend_at_middle = True,             # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        combine_upsample_fmaps = False,      # combine feature maps from all upsample blocks, used in unet squared successfully
        
        # last part
        init_conv_to_final_conv_residual = False,
        final_resnet_block = True,
        final_conv_kernel_size = 3,
        resize_mode = 'nearest',
        # ++
        CKeys = {'Debug_Level':0}, # for debug purpose: 0-silent mode
        
        # others: not used yet
        out_dim = None,                # output dim: not used
        num_image_tokens = 4,          # not used
        layer_attns_add_text_cond = True,   # whether to condition the self-attention blocks with the text embeddings, as described in Appendix D.3.1
        dropout = 0.,                  # not used, other part use their default dropout
        
    ):
        super().__init__()
        
        # 1. guide the setup by some checks
        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'
        if dim < 128:
            print_once(
                'The base dimension of your u-net should ideally be no smaller than 128, as recommended by a professional DDPM trainer https://nonint.com/2022/05/04/friends-dont-let-friends-train-small-diffusion-models/'
            )
        
        # 2. save locals to take care of some hyperparameters for cascading DDPM
        self._locals = locals()
        self._locals.pop('self', None) # remove self
        self._locals.pop('__class__', None)
        
        # ++ for debug level:
        self.CKeys = CKeys
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print ("Debug mode initiated...")
            print (f"Debug key: {self.CKeys}")
        
        # 3. determine dimensions
        self.channels = channels
        self.channels_out = default(channels_out, channels)
        # ++ for debug level:
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f".channels-in: {self.channels}")
            print (f".channels_out: {self.channels_out}")
        
        self.self_cond = self_cond
        self.lowres_cond = lowres_cond
        # optional image conditioning
        self.has_cond_image = cond_images_channels > 0
        self.cond_images_channels = cond_images_channels 
        
        # (1) in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        # (2) in self conditioning, one appends the predict x0 (x_start)
        # all of them have the same channel numbers, and same size
        # (3) also, consider image condition. This one may have independent channel #
        #
        # lowres_cond, self_cond are boolean
        init_channels = channels * (1 + int(lowres_cond) + int(self_cond))
        init_channels += cond_images_channels
        
        init_dim = default(init_dim, dim)
        # ++ for debug level:
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f"with low res image as condition: {self.lowres_cond}")
            print (f"with self-condition: {self.self_cond}")
            print (f"has cond image: {self.has_cond_image}")
            print ()
            print (f"So, init_channels: {init_channels}")
            print (f"init_dim: {init_dim}")
            
        # 4. initial convolution
        # in_dim: init_channels: include low-res-image, self-condition and cond_image
        # ++
        if init_cross_embed:
            self.init_conv = CrossEmbedLayer_OneD(
                init_channels, 
                dim_out = init_dim, 
                kernel_sizes = init_cross_embed_kernel_sizes, 
                stride = 1
            )
        else:
            self.init_conv = nn.Conv1d(
                init_channels, # in_channels
                init_dim,      # out_channels
                init_conv_kernel_size, 
                padding = init_conv_kernel_size // 2
            )
            # Conv2d: (N, C_in, H_in, W_in) -> (N, C_ou, H_ou, W_ou)
        # ++ for debug level:
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f"Create self.init_conv...")
            print (f"CrossEmbedLayer or Conv1d: {init_cross_embed}")
            print (f"stru: \n{self.init_conv}")
            
        # 5. middle layers
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:])) # pairs of in-ou embed dims
        
        # other conditions: position embed, and from diffusion time and text
        
        # 5.1 time conditioning
        cond_dim = default(cond_dim, dim)
        # 
        time_cond_dim = dim * 4 * (2 if lowres_cond else 1) # time_hiddens dim
        
        # embedding time for log(snr) noise from continuous version
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1
        
        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(
                sinu_pos_emb_input_dim, 
                time_cond_dim
            ),
            nn.SiLU()
        ) # out-dim: time_cond_dim
        
        self.to_time_cond = nn.Sequential(
            nn.Linear(
                time_cond_dim, 
                time_cond_dim
            )
        ) # out-dim: time_cond_dim
        
        # project to time tokens as well as time hiddens
        self.to_time_tokens = nn.Sequential(
            nn.Linear(
                time_cond_dim, 
                cond_dim * num_time_tokens
            ),
            Rearrange('b (r d) -> b r d', r = num_time_tokens)
        ) # out-dim: (batch, num_time_tokens, cond_dim)
        
        # ++ for debug level:
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f"time_cond_dim: {time_cond_dim}")
            print (f"output dim of sinu_pos_emb: {sinu_pos_emb_input_dim}")
            print (f"output dim of .to_time_hiddens: {time_cond_dim}")
            print (f"output dim of .to_time_cond: {time_cond_dim}")
            print (f"output dim of .to_time_tokens: [b, {num_time_tokens}, {cond_dim}]")
        
        # 5.2 for lowres_cond
        if lowres_cond:
            self.to_lowres_time_hiddens = nn.Sequential(
                LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim),
                nn.Linear(learned_sinu_pos_emb_dim + 1, time_cond_dim),
                nn.SiLU()
            ) # out-dim: time_cond_dim

            self.to_lowres_time_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim)
            )

            self.to_lowres_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                Rearrange('b (r d) -> b r d', r = num_time_tokens)
            )
            # ++ for debug level:
            if self.CKeys['Debug_Level']==UNet_Init_Level:
                print (f".to_lowres_time_hiddens: {self.to_lowres_time_hiddens}")
                print (f".to_lowres_time_cond: {self.to_lowres_time_cond}")
                print (f".to_lowres_time_tokens: {self.to_lowres_time_tokens}")
        
        # 6. normalization for conditions
        self.norm_cond = nn.LayerNorm(cond_dim)
        # ++ for debug level:
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f".norm_cond: {self.norm_cond}")
            
        # 7. text encoding conditioning (optional)
        self.text_to_cond = None
        # ++
        self.text_embed_dim = text_embed_dim
        if cond_on_text:
            assert exists(text_embed_dim), 'text_embed_dim must be given to the unet if cond_on_text is True'
            self.text_to_cond = nn.Linear(
                text_embed_dim, 
                cond_dim
            )
            # ++
            if self.CKeys['Debug_Level']==UNet_Init_Level:
                print (f"mapping from text emb into unet cond_dim")
                print (f".text_to_cond: {self.text_to_cond}")  
                
        # 7.1 finer control over whether to condition on text encodings
        self.cond_on_text = cond_on_text
        # ++
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f".cond_on_text: {self.cond_on_text}")
        
        # attention pooling
        
        self.attn_pool = PerceiverResampler(
            dim = cond_dim, 
            depth = 2, # not sure why pick this shallow one
            dim_head = attn_dim_head, 
            heads = attn_heads, 
            num_latents = attn_pool_num_latents,
            # ++
            max_seq_len = max_text_len, # 512,
            
        ) if attn_pool_text else None
        # to be opened up
        
        # for classifier free guidance
        self.max_text_len = max_text_len
        # 
        self.null_text_embed = nn.Parameter(
            torch.randn(1, max_text_len, cond_dim)
        )
        self.null_text_hidden = nn.Parameter(
            torch.randn(1, time_cond_dim)
        )
        # ++ for debug level:
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f"self.max_text_len: {self.max_text_len}")
            print (f"self.null_text_embed.shape: {self.null_text_embed.shape}")
            print (f"self.null_text_hidden.shape: {self.null_text_hidden.shape}")
                
        # for non-attention based text conditioning at all points in the network
        # where time is also conditioned
        self.to_text_non_attn_cond = None
        
        if cond_on_text:
            self.to_text_non_attn_cond = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, time_cond_dim)
            ) # out-dim: time_cond_dim
        # ++
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f".to_text_non_attn_cond: \n{self.to_text_non_attn_cond}")
        
        # 8. attention related params
        
        attn_kwargs = dict(
            heads = attn_heads, 
            dim_head = attn_dim_head,
        )
        num_layers = len(in_out)
        
        # resnet block klass
        
        num_resnet_blocks = cast_tuple(
            num_resnet_blocks, 
            num_layers
        )
        
        resnet_klass = partial(
            ResnetBlock_OneD, 
            **attn_kwargs,
        )
        
        layer_attns = cast_tuple(
            layer_attns, num_layers
        )
        layer_attns_depth = cast_tuple(
            layer_attns_depth, num_layers
        )
        layer_cross_attns = cast_tuple(
            layer_cross_attns, num_layers
        )
        
        use_linear_attn = cast_tuple(
            use_linear_attn, num_layers
        )
        use_linear_cross_attn = cast_tuple(
            use_linear_cross_attn, num_layers
        )
        # ++
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f"CHeck")
            print (f"  num_resnet_blocks: \n{num_resnet_blocks}")
            print (f"  layer_attns: \n{layer_attns}")
        
        assert all(
            [
                layers == num_layers for layers in list(
                    map(len, (layer_attns, layer_cross_attns))
                )
            ]
        )
        
        # downsample klass
        
        downsample_klass = Downsample_OneD
        
        if cross_embed_downsample:
            downsample_klass = partial(
                CrossEmbedLayer_OneD, 
                kernel_sizes = cross_embed_downsample_kernel_sizes
            )
            # ++
            if self.CKeys['Debug_Level']==UNet_Init_Level:
                print (f"Arg for downsample_class: {cross_embed_downsample_kernel_sizes}")
        # 
        # initial resnet block (for memory efficient unet)
        
        self.init_resnet_block = resnet_klass(
            init_dim, init_dim, 
            time_cond_dim = time_cond_dim, 
            use_gca = use_global_context_attn
        ) if memory_efficient else None
        
        # scale for resnet skip connections
        
        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)
        # ++
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f".init_resnet_block: {self.init_resnet_block}")
            print (f".skip_connect_scale: {self.skip_connect_scale}")
            
        # layers: up and down
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) # layers inside the encoder/decoder of UNet
        
        layer_params = [
            num_resnet_blocks, 
            layer_attns, 
            layer_attns_depth, 
            layer_cross_attns, 
            use_linear_attn, 
            use_linear_cross_attn
        ]
        reversed_layer_params = list(map(reversed, layer_params))
        # ++
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f"On augments, layer_params: \n{layer_params}")
            # explain
            print (f"On UNet setup for all blocks: ")
            print (f"  Blocks in UNet: {len(in_out)}")
            print (f"  in-ou dims: {in_out}")
            print (f"  resnet_block #: {num_resnet_blocks}")
            print (f"  add attn or not: {layer_attns}")
            print (f"  attn depth: {layer_attns_depth}")
            print (f"  use cross attns: {layer_cross_attns}")
            print (f"  use linear attns: {use_linear_attn}")
            print (f"  use linear cross attn: {use_linear_cross_attn}")
            
            # print (f"reversed_layer_params: \n{reversed_layer_params}")
        
        # 9. downsampling layers
        
        skip_connect_dims = [] # keep track of skip connection dimensions

        for ind, (
            (dim_in, dim_out), 
            layer_num_resnet_blocks, 
            layer_attn, 
            layer_attn_depth, 
            layer_cross_attn, 
            layer_use_linear_attn, 
            layer_use_linear_cross_attn
        ) in enumerate(zip(in_out, *layer_params)):
            # ++
            if self.CKeys['Debug_Level']==UNet_Init_Level:
                print (f"\n===========================================")
                print (f"For Downsample Block {ind}")
                print (f"  dims in & out: {dim_in}, {dim_out}")
                print (f"  resnet block: {layer_num_resnet_blocks}")
                print (f"  attn layer: {layer_attn}")
                print (f"  attn layer depth: {layer_attn_depth}")
                print (f"  cross attn: {layer_cross_attn}")
                print (f"  linear attn: {layer_use_linear_attn}")
                print (f"  linear cross attn: {layer_use_linear_cross_attn}")
                
            
            is_last = ind >= (num_resolutions - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            if layer_attn: # Attention > Linear Attension
                transformer_block_klass = TransformerBlock_OneD
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet

            pre_downsample = None
            
            if memory_efficient:
                pre_downsample = downsample_klass(dim_in, dim_out)
                current_dim = dim_out
            

            skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet
            
            post_downsample = None
            if not memory_efficient:
                post_downsample = downsample_klass(
                    current_dim, 
                    dim_out = dim_out,
                ) if not is_last else Parallel(
                    # --
                    # nn.Conv2d(dim_in, dim_out, 3, padding = 1), 
                    # nn.Conv2d(dim_in, dim_out, 1)
                    # ++
                    nn.Conv1d(dim_in, dim_out, 3, padding = 1), 
                    nn.Conv1d(dim_in, dim_out, 1),
                )
            # dimension reduction happen in this layer, so look into it
            # may consider change dim_out == dim_in so no reduction is used
            # ++
            if self.CKeys['Debug_Level']==UNet_Init_Level:
                print (f" Dimension reduction layer: post_downsample\n{post_downsample} ")

            self.downs.append(
                nn.ModuleList(
                    [
                        # ============================================
                        # Multi-scale convolution on embeddin
                        pre_downsample,
                        # ============================================
                        # init_resnet_block: ResnetBlock
                        # with cross-attn for conditioning
                        resnet_klass(
                            current_dim, 
                            current_dim, 
                            time_cond_dim = time_cond_dim,
                            # 
                            cond_dim = layer_cond_dim, 
                            linear_attn = layer_use_linear_cross_attn, 
                            
                        ),
                        # ============================================
                        # ResnetBlocks: with global context attn
                        # 
                        nn.ModuleList(
                            [
                                ResnetBlock_OneD(
                                    current_dim, 
                                    current_dim, 
                                    time_cond_dim = time_cond_dim,
                                    # 
                                    use_gca = use_global_context_attn
                                ) for _ in range(layer_num_resnet_blocks)
                            ]
                        ),
                        # ============================================
                        # Attention block: via conditioning
                        # 
                        transformer_block_klass(
                            dim = current_dim, 
                            depth = layer_attn_depth, 
                            ff_mult = ff_mult, 
                            context_dim = cond_dim, 
                            **attn_kwargs
                        ),
                        # ============================================
                        post_downsample
                        # ============================================
                    ]
                )
            )

        # middle layers
        
        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock_OneD(
            mid_dim, mid_dim, 
            cond_dim = cond_dim, 
            time_cond_dim = time_cond_dim
        )
        self.mid_attn = TransformerBlock_OneD(
            mid_dim, 
            depth = layer_mid_attns_depth, 
            **attn_kwargs
        ) if attend_at_middle else None
        self.mid_block2 = ResnetBlock_OneD(
            mid_dim, mid_dim, 
            cond_dim = cond_dim, 
            time_cond_dim = time_cond_dim
        )
        
        # upsample klass

        upsample_klass = Upsample_OneD if not pixel_shuffle_upsample else PixelShuffleUpsample_OneD

        # upsampling layers

        upsample_fmap_dims = []

        for ind, (
            (dim_in, dim_out), 
            layer_num_resnet_blocks, 
            layer_attn, 
            layer_attn_depth, 
            layer_cross_attn, 
            layer_use_linear_attn, 
            layer_use_linear_cross_attn
        ) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            # ++
            if self.CKeys['Debug_Level']==UNet_Init_Level:
                print (f"\n===========================================")
                print (f"For Upsample Block {ind}")
                print (f"  dims in $ out: {dim_in}, {dim_out}")
                print (f"  resnet block: {layer_num_resnet_blocks}")
                print (f"  attn layer: {layer_attn}")
                print (f"  attn layer depth: {layer_attn_depth}")
                print (f"  cross attn: {layer_cross_attn}")
                print (f"  linear attn: {layer_use_linear_attn}")
                print (f"  linear cross attn: {layer_use_linear_cross_attn}")
            
            is_last = ind == (len(in_out) - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            if layer_attn:
                transformer_block_klass = TransformerBlock_OneD
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            skip_connect_dim = skip_connect_dims.pop()

            upsample_fmap_dims.append(dim_out)

            self.ups.append(
                nn.ModuleList(
                    [
                        # ============================================
                        resnet_klass(
                            dim_out + skip_connect_dim, 
                            dim_out, 
                            cond_dim = layer_cond_dim, 
                            linear_attn = layer_use_linear_cross_attn, 
                            time_cond_dim = time_cond_dim
                        ),
                        # ============================================
                        nn.ModuleList(
                            [
                                ResnetBlock_OneD(
                                    dim_out + skip_connect_dim, 
                                    dim_out, 
                                    time_cond_dim = time_cond_dim, 
                                    use_gca = use_global_context_attn
                                ) for _ in range(layer_num_resnet_blocks)]
                        ),
                        # ============================================
                        transformer_block_klass(
                            dim = dim_out, 
                            depth = layer_attn_depth, 
                            ff_mult = ff_mult, 
                            context_dim = cond_dim, 
                            **attn_kwargs
                        ),
                        # ============================================
                        upsample_klass(
                            dim_out, 
                            dim_in
                        ) if not is_last or memory_efficient else Identity()
                        # ============================================
                    ]
                )
            )

        # whether to combine feature maps from all upsample blocks before final resnet block out

        self.upsample_combiner = UpsampleCombiner_OneD(
            dim = dim,
            enabled = combine_upsample_fmaps,
            dim_ins = upsample_fmap_dims,
            dim_outs = dim
        )
        # ++
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f".upsample_combiner: {self.upsample_combiner}")
            
        # whether to do a final residual from initial conv to the final resnet block out
        
        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = self.upsample_combiner.dim_out + (
            dim if init_conv_to_final_conv_residual else 0
        )
        # ++
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f".init_conv_to_final_conv_residual: {self.init_conv_to_final_conv_residual}")
            
        # final optional resnet block and convolution out
        
        self.final_res_block = ResnetBlock_OneD(
            final_conv_dim, 
            dim, 
            time_cond_dim = time_cond_dim, 
            use_gca = True
        ) if final_resnet_block else None
        # ++
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f"self.final_res_block: {self.final_res_block}")
            
        final_conv_dim_in = dim if final_resnet_block else final_conv_dim
        final_conv_dim_in += (channels if lowres_cond else 0)
        
        self.final_conv = nn.Conv1d(
            final_conv_dim_in, 
            self.channels_out, 
            final_conv_kernel_size, 
            padding = final_conv_kernel_size // 2
        )
        # ++
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f"self.final_conv: \n{self.final_conv}")
        
        zero_init_(self.final_conv)
        
        # resize mode
        
        self.resize_mode = resize_mode
        # ++
        if self.CKeys['Debug_Level']==UNet_Init_Level:
            print (f"self.resize_mode: {self.resize_mode}")
    # //////////////////////////////////////////////////////////
    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        text_embed_dim,
        channels,
        channels_out,
        cond_on_text
    ):
        if lowres_cond == self.lowres_cond and \
            channels == self.channels and \
            cond_on_text == self.cond_on_text and \
            text_embed_dim == self._locals['text_embed_dim'] and \
            channels_out == self.channels_out:
            return self

        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            text_embed_dim = text_embed_dim,
            channels = channels,
            channels_out = channels_out,
            cond_on_text = cond_on_text
        )

        return self.__class__(**{**self._locals, **updated_kwargs})
    
    # //////////////////////////////////////////////////////////
    # methods for returning the full unet config as well as its parameter state
    def to_config_and_state_dict(self):
        return self._locals, self.state_dict()
    
    # //////////////////////////////////////////////////////////
    # class method for rehydrating the unet from its config and state dict
    @classmethod
    def from_config_and_state_dict(klass, config, state_dict):
        unet = klass(**config)
        unet.load_state_dict(state_dict)
        return unet
    
    # //////////////////////////////////////////////////////////
    # methods for persisting unet to disk
    def persist_to_file(self, path):
        path = Path(path)
        path.parents[0].mkdir(exist_ok = True, parents = True)

        config, state_dict = self.to_config_and_state_dict()
        pkg = dict(config = config, state_dict = state_dict)
        torch.save(pkg, str(path))
    
    # //////////////////////////////////////////////////////////
    # class method for rehydrating the unet from file saved with `persist_to_file`
    @classmethod
    def hydrate_from_file(klass, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))

        assert 'config' in pkg and 'state_dict' in pkg
        config, state_dict = pkg['config'], pkg['state_dict']

        return Unet.from_config_and_state_dict(config, state_dict)
    
    # //////////////////////////////////////////////////////////
    # forward with classifier free guidance
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    
    # //////////////////////////////////////////////////////////
    def forward(
        self,
        x,
        time,
        *,
        # 
        self_cond = None,
        # 
        lowres_cond_img = None,
        lowres_noise_times = None,
        # 
        cond_images = None,
        # 
        text_embeds = None,
        text_mask = None,
        # 
        cond_drop_prob = 0.
    ):
        # ++
        if self.CKeys['Debug_Level']==UNet_Forw_Level:
            print (f"========================================")
            print (f"In Unet_OneD.forward()...")
        # =====================================================
        # 0. prepare
        
        batch_size, device = x.shape[0], x.device
        # ++
        if self.CKeys['Debug_Level']==UNet_Forw_Level:
            print (f"batch size: {batch_size}")
            print (f"device: {device}")
            print (f"Initial, x.shape: {x.shape}")
            # (b, c, H, W)
            
        # condition on self
        
        if self.self_cond:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, self_cond), dim = 1)
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"After self_cond, x.shape: {x.shape}")
                # (b, c+C_self, H, W)
                
        #  add low resolution conditioning, if presented
        
        assert not (self.lowres_cond and not exists(lowres_cond_img)), 'low resolution conditioning image must be present'
        assert not (self.lowres_cond and not exists(lowres_noise_times)), 'low resolution conditioning noise time must be present'
        
        if exists(lowres_cond_img):
            x = torch.cat(
                (x, lowres_cond_img),
                dim = 1,
            )
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"After low res image conditioning, x.shape: {x.shape}")
                # (b, c+C_self+C_lowres, H, W)
                
        # condition on input image
        
        assert not (self.has_cond_image ^ exists(cond_images)), 'you either requested to condition on an image on the unet, but the conditioning image is not supplied, or vice versa'
        
        if exists(cond_images):
            assert cond_images.shape[1] == self.cond_images_channels, 'the number of channels on the conditioning image you are passing in does not match what you specified on initialiation of the unet'
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"Via input, cond_images.shape: {cond_images.shape}")
            cond_images = resize_2d_image_to(
                cond_images, 
                x.shape[-1], 
                mode = self.resize_mode
            )
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"After resize, cond_images.shape: {cond_images.shape}")
                
            # match the image based on the last dimension
            x = torch.cat(
                (cond_images, x), dim = 1
            )
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"After cond_images, x.shape: {x.shape}")
                # (b, C_cond_img + c+C_self+C_lowres, H, W)
                
        # =====================================================
        # 1. initial convolution
        # ++
        if self.CKeys['Debug_Level']==UNet_Forw_Level:
            print (f"Before x.shape: {x.shape}")
        x = self.init_conv(x)
        # ++
        if self.CKeys['Debug_Level']==UNet_Forw_Level:
            print (f".init_conv: \n{self.init_conv}")
            print (f"Note: apply conv with different kernel and cat all the output)")
            print (f"After .init_conv(x), x.shape: {x.shape}")
            
        # init conv residual
        
        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"init_conv_residual.shape: {init_conv_residual.shape}")
                # should be the same as x
                
        # time conditioning
        
        time_hiddens = self.to_time_hiddens(time)
        # ++
        if self.CKeys['Debug_Level']==UNet_Forw_Level:
            print (f"On diffusion time conditioning:")
            print (f"in time.shape: {time.shape}")
            print (f".to_time_hiddens: {self.to_time_hiddens}")
            print (f"out time_hiddens.shape: {time_hiddens.shape}")
        # time_hiddens: like a learnable postion embedding
        
        # derive time tokens: from time_hiddens: TBU, what is this for
        # derive time conditions: from time_hiddens
        
        time_tokens = self.to_time_tokens(time_hiddens)
        t           = self.to_time_cond  (time_hiddens)
        # ++
        if self.CKeys['Debug_Level']==UNet_Forw_Level:
            print (f"self.to_time_tokens: {self.to_time_tokens}")
            print (f"ou: time_tokens.shape: {time_tokens.shape}")
            print (f"self.to_time_cond: {self.to_time_cond}")
            print (f"ou: t.shape: {t.shape}")
            
        # add lowres time conditioning to time hiddens
        # and add lowres time tokens along sequence dimension for attention
        
        if self.lowres_cond:
            lowres_time_hiddens = self.to_lowres_time_hiddens(lowres_noise_times)
            lowres_time_tokens = self.to_lowres_time_tokens(lowres_time_hiddens)
            lowres_t = self.to_lowres_time_cond(lowres_time_hiddens)
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f".to_lowres_time_hiddens(lowres_noise_times).shape: {lowres_time_hiddens.shape}")
                print (f".to_lowres_time_tokens(lowres_time_hiddens).shape: {lowres_time_tokens.shape}")
                print (f".to_lowres_time_cond(lowres_time_hiddens).shape: {lowres_t.shape}")
            
            t = t + lowres_t
            time_tokens = torch.cat(
                (time_tokens, lowres_time_tokens), 
                dim = -2
            )
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"time_tokens.shape: {time_tokens.shape}")
                
        # text conditioning
        
        text_tokens = None
        
        if exists(text_embeds) and self.cond_on_text:
            
            # conditional dropout: based on a given probability
            
            text_keep_mask = prob_mask_like(
                (batch_size,), 
                1 - cond_drop_prob, 
                device = device
            )
            
            text_keep_mask_embed = rearrange(text_keep_mask, 'b -> b 1 1')
            text_keep_mask_hidden = rearrange(text_keep_mask, 'b -> b 1')
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"text_keep_mask: {text_keep_mask}") # (b)
                print (f"text_keep_mask_embed.shape: {text_keep_mask_embed.shape}")
                
            # calculate text embeds
            
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"text_embeds.shape: {text_embeds.shape}")
            text_tokens = self.text_to_cond(text_embeds) 
            # Linear(in_features=text_emb_dim, out_features=cond_emb_dim, bias=True)
            
            
            text_tokens = text_tokens[:, :self.max_text_len]
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"self.text_to_cond: \n{self.text_to_cond}")
                print (f"ou: text_tokens.shape: {text_tokens.shape}")
            
            if exists(text_mask):
                # chop the length of text_mask
                text_mask = text_mask[:, :self.max_text_len] # assume shape (b, seq_len)
                
            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len
            
            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))
                
            if exists(text_mask):
                if remainder > 0:
                    text_mask = F.pad(text_mask, (0, remainder), value = False)

                text_mask = rearrange(text_mask, 'b n -> b n 1')
                # now, merge the keeping mask, and the input mask
                text_keep_mask_embed = text_mask & text_keep_mask_embed
            
            null_text_embed = self.null_text_embed.to(text_tokens.dtype) # for some reason pytorch AMP not working
            
            # apply the mask
            text_tokens = torch.where(
                text_keep_mask_embed, # condition
                text_tokens,          # keep if true
                null_text_embed,       # else
            )
            #  (b, max_text_len, cond_dim)
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"After masking: ")
                print (f"ou: text_tokens.shape: {text_tokens.shape}")
            
            if exists(self.attn_pool):
                text_tokens = self.attn_pool(text_tokens)
                # ++
                if self.CKeys['Debug_Level']==UNet_Forw_Level:
                    print (f"Apply self.attn_pool: {self.attn_pool}")
                    print (f"Ou: text_tokens.shape: {text_tokens.shape}")
             
            # extra non-attention conditioning by projecting and then summing text embeddings to time
            # termed as text hiddens
            mean_pooled_text_tokens = text_tokens.mean(dim = -2)
            text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"text_tokens=>mean_pooled_text_tokens.shape: {mean_pooled_text_tokens.shape}")
                print (f"pass into self.to_text_non_attn_cond: \n{self.to_text_non_attn_cond}")
                print (f"ou: text_hiddens.shape: {text_hiddens.shape}")
                
            null_text_hidden = self.null_text_hidden.to(t.dtype)
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"null_text_hidden.shape: {null_text_hidden.shape}")
            # apply masking
            text_hiddens = torch.where(
                text_keep_mask_hidden,
                text_hiddens,
                null_text_hidden
            )
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"After masking, text_hiddens: \n{text_hiddens.shape}")
            
            t = t + text_hiddens
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"Merge time and text_hiddens:")
                print (f"As t+text_hiddens -> t.shape: {t.shape}")
            
        # main conditioning tokens (c)
        # 
        if not exists(text_tokens):
            c = time_tokens
        else:
            c = torch.cat(
                (time_tokens, text_tokens), 
                dim = -2
            )
        # ++
        if self.CKeys['Debug_Level']==UNet_Forw_Level:
            print (f"Merge time_tokens and text_tokens,")
            print (f"in: time_tokens.shape: {time_tokens.shape}")
            print (f"in: text_tokens.shape: {text_tokens.shape}")
            print (f"ou: cat(,) -> c.shape: {c.shape}")
        
        # normalize conditioning tokens
        c = self.norm_cond(c)
        
        # initial resnet block (for memory efficient unet)
        
        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                # this one only activate if memory_efficient is true
                print (f".init_resnet_block(x,t)->x.shape: {x.shape}")
            
        # go through the layers of the unet, down, middle and up
        # Note, there is only one UNet here

        hiddens = []
        
        # ++
        if self.CKeys['Debug_Level']==UNet_Forw_Level:
            print (f"Next, get into self.downs loop:")
            print (f"-------------------------------")
            
        for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs:
            # 1. pre downsample:
            if exists(pre_downsample):
                x = pre_downsample(x) # Multi-scale convo parallel
                # ++
                if self.CKeys['Debug_Level']==UNet_Forw_Level:
                    print (f"pre_downsample(x)->x.shape: {x.shape}")
            
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"x, t, c.shape: {x.shape}, {t.shape}, {c.shape}")
            if Test_Debug_Level_2 == 1:
                print (f"init_block: \n{init_block}")
            x = init_block(x, t, c) # ResNet block with conditioning and timining
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"init_block(x, t, c)->x.shape: {x.shape}")
                
            for resnet_block in resnet_blocks:
                x = resnet_block(x, t) # ResNet block with timing without conditioning
                hiddens.append(x)
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"resnet_block(x, t)->x.shape: {x.shape}")

            x = attn_block(x, c) # attn blocks with conditioning
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"attn_block(x, c)->x.shape: {x.shape}")
            hiddens.append(x)

            if exists(post_downsample):
                x = post_downsample(x) # Multi-scale convo parallel
                # ++
                if self.CKeys['Debug_Level']==UNet_Forw_Level:
                    print (f"post_downsample(x)->x.shape: {x.shape}")
        # =============================================
        # middle blocks
        x = self.mid_block1(x, t, c) # RseNet with t and c
        # ++
        if self.CKeys['Debug_Level']==UNet_Forw_Level:
            print (f"mid_block1(x,t,c)->x.shape: {x.shape}")

        if exists(self.mid_attn):
            x = self.mid_attn(x) #  attn without t or c
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"mid_attn(x)->x.shape: {x.shape}")

        x = self.mid_block2(x, t, c)
        # ++
        if self.CKeys['Debug_Level']==UNet_Forw_Level:
            print (f"mid_block2(x,t,c)->x.shape: {x.shape}")
            
        # ============================================
        # upsample part
            
        add_skip_connection = lambda x: torch.cat(
            (
                x, hiddens.pop() * self.skip_connect_scale
            ), 
            dim = 1
        )

        up_hiddens = []
        
        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            x = add_skip_connection(x)
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"add_skip_connection(x)->x.shape: {x.shape}")
                
            x = init_block(x, t, c) # ResNet with t and c
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"init_block(x,t,c)->x.shape: {x.shape}")
            

            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t) # ResNet with t
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"(resnet_block(x,t)->x.shape: {x.shape})")

            x = attn_block(x, c) # Attention with c
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"attn_block(x,c)->x.shape: {x.shape}")
                
            up_hiddens.append(x.contiguous())
            x = upsample(x) # ResNet
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"upsample(x)->x.shape: {x.shape}")

        # whether to combine all feature maps from upsample blocks

        x = self.upsample_combiner(x, up_hiddens)
        # ++
        if self.CKeys['Debug_Level']==UNet_Forw_Level:
            print (f"upsample_combiner(x,up_hiddens)->x.shape: {x.shape}")

        # final top-most residual if needed

        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim = 1)
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"cat(x, init_conv_residual)->x.shape: {x.shape}")

        if exists(self.final_res_block):
            x = self.final_res_block(x, t)
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"final_res_block(x, t)->x.shape: {x.shape}")

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)
            # ++
            if self.CKeys['Debug_Level']==UNet_Forw_Level:
                print (f"cat(x, lowres_cond_img)->x.shape: {x.shape}")
                
        x = self.final_conv(x)
        # ++
        if self.CKeys['Debug_Level']==UNet_Forw_Level:
            print (f".final_conv(x)->x.shape: {x.shape}")

        return x
