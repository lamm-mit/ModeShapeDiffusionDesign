import math
from math import sqrt
from random import random
from functools import partial, wraps
from contextlib import contextmanager, nullcontext
from typing import List, Union
from collections import namedtuple
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel
import torchvision.transforms as T

import kornia.augmentation as K

from einops import rearrange, repeat, reduce

from torch.special import expm1

# --
# from imagen_pytorch_study.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

from VibeGen.imagen_x_unet_pytorch import (
    # GaussianDiffusionContinuousTimes,
    Unet_OneD, # Unet_TwoD,
    NullUnet,
    # first,
    exists,
    # identity,
    # maybe,
    default,
    cast_tuple,
    # cast_uint8_images_to_float,
    # eval_decorator,
    # pad_tuple_to_length,
    resize_2d_image_to,
    # calc_all_frame_dims,
    # safe_get_tuple_index,
    # right_pad_dims_to,
    # module_device,
    # normalize_neg_one_to_one,
    # unnormalize_zero_to_one,
    # compact,
    # maybe_transform_dict_key
    Unet3D,
    resize_video_to,
    scale_video_time,
)

# --
# from imagen_pytorch_study.imagen_video import (
#     Unet3D,
#     resize_video_to,
#     scale_video_time
# )
# 
# ==========================================================
Imagen_Init_Level = 1 # for initialization
Imagen_Forw_Level = 1 # for forward func
Imagen_Samp_Level = 1 # for forward func
# ===========================================================
def identity(t, *args, **kwargs):
    return t

def divisible_by(numer, denom):
    return (numer % denom) == 0

def first(arr, d = None):
    if len(arr) == 0:
        return d
    return arr[0]

def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)
    return inner


def cast_uint8_images_to_float(images):
    '''
    1. change format
    2. normalize into (0, 1)
    ? may need to adjust for 1D case: 
    No. It is not triggered as long as images is not uint8
    '''
    if not images.dtype == torch.uint8:
        return images
    return images / 255

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def pad_tuple_to_length(t, length, fillvalue = None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue,) * remain_length))

def calc_all_frame_dims(
    downsample_factors: List[int],
    frames
):
    if not exists(frames):
        return (tuple(),) * len(downsample_factors)

    all_frame_dims = []

    for divisor in downsample_factors:
        assert divisible_by(frames, divisor)
        all_frame_dims.append((frames // divisor,))

    return all_frame_dims

def safe_get_tuple_index(tup, index, default = None):
    if len(tup) <= index:
        return default
    return tup[index]

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def module_device(module):
    return next(module.parameters()).device

# image normalization functions
# ddpms expect images to be in the range of -1 to 1
# map from (0,1)->(-1,1)
def normalize_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5

def compact(input_dict):
    return {key: value for key, value in input_dict.items() if exists(value)}

def maybe_transform_dict_key(input_dict, key, fn):
    if key not in input_dict:
        return input_dict

    copied_dict = input_dict.copy()
    copied_dict[key] = fn(copied_dict[key])
    return copied_dict

# tensor helpers

def log(t, eps: float = 1e-12):
    return torch.log(t.clamp(min = eps))

# 
# ===========================================================
# ===========================================================
# ===========================================================
# main class: ElucidatedImagen
# ===========================================================
# ===========================================================
# ===========================================================
#
# on diffusion scheduler
# 
# gaussian diffusion with continuous time helper functions and classes
# large part of this was thanks to @crowsonkb at https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

@torch.jit.script
def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

@torch.jit.script
def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5) # not sure if this accounts for beta being clipped to 0.999 in discrete version

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))
# 
class GaussianDiffusionContinuousTimes(nn.Module):
    def __init__(
        self, 
        *, 
        noise_schedule, 
        timesteps = 1000,
    ):
        super().__init__()

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.num_timesteps = timesteps

    def get_times(
        self, 
        batch_size, 
        noise_level, 
        *, 
        device
    ):
        return torch.full(
            (batch_size,), 
            noise_level, 
            device = device, 
            dtype = torch.float32
        )

    def sample_random_times(
        self, 
        batch_size, 
        *, 
        device
    ):
        return torch.zeros(
            (batch_size,), 
            device = device
        ).float().uniform_(0, 1)

    def get_condition(self, times):
        return maybe(self.log_snr)(times)

    def get_sampling_timesteps(
        self, 
        batch, 
        *, 
        device
    ):
        times = torch.linspace(
            1., 
            0., 
            self.num_timesteps + 1, 
            device = device
        )
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    def q_posterior(
        self, 
        x_start, 
        x_t, 
        t, 
        *, 
        t_next = None
    ):
        t_next = default(
            t_next, 
            lambda: (t - 1. / self.num_timesteps).clamp(min = 0.)
        )

        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(
            partial(right_pad_dims_to, x_t), 
            (log_snr, log_snr_next)
        )

        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)

        # following (eq. 33)
        posterior_variance = (sigma_next ** 2) * c
        posterior_log_variance_clipped = log(posterior_variance, eps = 1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(
        self, 
        x_start, 
        t, 
        noise = None
    ):
        dtype = x_start.dtype

        if isinstance(t, float):
            batch = x_start.shape[0]
            t = torch.full((batch,), t, device = x_start.device, dtype = dtype)

        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr(t).type(dtype)
        log_snr_padded_dim = right_pad_dims_to(x_start, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)

        return alpha * x_start + sigma * noise, log_snr, alpha, sigma

    def q_sample_from_to(
        self, 
        x_from, 
        from_t, 
        to_t, 
        noise = None
    ):
        shape, device, dtype = x_from.shape, x_from.device, x_from.dtype
        batch = shape[0]

        if isinstance(from_t, float):
            from_t = torch.full((batch,), from_t, device = device, dtype = dtype)

        if isinstance(to_t, float):
            to_t = torch.full((batch,), to_t, device = device, dtype = dtype)

        noise = default(noise, lambda: torch.randn_like(x_from))

        log_snr = self.log_snr(from_t)
        log_snr_padded_dim = right_pad_dims_to(x_from, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)

        log_snr_to = self.log_snr(to_t)
        log_snr_padded_dim_to = right_pad_dims_to(x_from, log_snr_to)
        alpha_to, sigma_to =  log_snr_to_alpha_sigma(log_snr_padded_dim_to)

        return x_from * (alpha_to / alpha) + noise * (sigma_to * alpha - sigma * alpha_to) / alpha

    def predict_start_from_v(self, x_t, t, v):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return alpha * x_t - sigma * v

    def predict_start_from_noise(self, x_t, t, noise):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / alpha.clamp(min = 1e-8)
    
# ===========================================================
# constants

Hparams_fields = [
    'num_sample_steps',
    'sigma_min',
    'sigma_max',
    'sigma_data',
    'rho',
    'P_mean',
    'P_std',
    'S_churn',
    'S_tmin',
    'S_tmax',
    'S_noise'
]

Hparams = namedtuple('Hparams', Hparams_fields)


# ===========================================================
# ===========================================================
# add for OneD data format
#
class ElucidatedImagen_OneD(nn.Module):
    def __init__(
        self,
        # 1. unets: many setups of UNet will be passed on via UNet itself
        unets,
        *,
        channels = 3,
        # 2. in-output image size
        image_sizes,                                # for cascading ddpm, image size at each stage
        # 3. on text conditioning
        text_encoder_name = None, # TBU: DEFAULT_T5_NAME,
        text_embed_dim = None,
        cond_drop_prob = 0.1,
        condition_on_text = True,
        # 
        random_crop_sizes = None,
        resize_mode = 'nearest',
        temporal_downsample_factor = 1,
        resize_cond_video_frames = True,
        lowres_sample_noise_level = 0.2,            # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
        per_sample_random_aug_noise_level = False,  # unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
        auto_normalize_img = True,                  # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
        dynamic_thresholding = True,
        dynamic_thresholding_percentile = 0.95,     # unsure what this was based on perusal of paper
        only_train_unet_number = None,
        lowres_noise_schedule = 'linear',
        num_sample_steps = 32,                      # number of sampling steps
        sigma_min = 0.002,                          # min noise level
        sigma_max = 80,                             # max noise level
        sigma_data = 0.5,                           # standard deviation of data distribution
        rho = 7,                                    # controls the sampling schedule
        P_mean = -1.2,                              # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,                                # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,                               # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
        # ++
        CKeys = {'Debug_Level':0}, # for debug purpose: 0--silence mode
    ):
        super().__init__()
        
        # ++ for debug
        self.CKeys = CKeys
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print ("Debug mode: Initialization of EImagen...\n")
        
        self.only_train_unet_number = only_train_unet_number
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".only_train_unet_number: {self.only_train_unet_number}")

        # conditioning hparams

        self.condition_on_text = condition_on_text
        self.unconditional = not condition_on_text
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".condition_on_text: {self.condition_on_text}")
            print (f".unconditional: {self.unconditional}")

        # channels

        self.channels = channels
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".channels: {self.channels}")

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet

        unets = cast_tuple(unets)
        num_unets = len(unets)

        # randomly cropping for upsampler training

        self.random_crop_sizes = cast_tuple(random_crop_sizes, num_unets)
        assert not exists(first(self.random_crop_sizes)), 'you should not need to randomly crop image during training for base unet, only for upsamplers - so pass in `random_crop_sizes = (None, 128, 256)` as example'
        # may get rid of this when moving to 1d case
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".random_crop_sizes: {self.random_crop_sizes}")

        # lowres augmentation noise schedule

        self.lowres_noise_schedule = GaussianDiffusionContinuousTimes(
            noise_schedule = lowres_noise_schedule
        )
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".lowres_noise_schedule: {self.lowres_noise_schedule}")

        # get text encoder

        self.text_encoder_name = text_encoder_name
        # --: a dull one is enough
        # self.text_embed_dim = default(text_embed_dim, lambda: get_encoded_dim(text_encoder_name))
        # ++
        self.text_embed_dim = text_embed_dim
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".text_encoder_name: {self.text_encoder_name}")
            print (f".text_embed_dim: {self.text_embed_dim}")

        # -- text channel is not updated yet
        # self.encode_text = partial(t5_encode_text, name = text_encoder_name)
        # ++: TBU if needed
        self.encode_text = None
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".encode_text: {self.encode_text}")

        # construct unets

        self.unets = nn.ModuleList([])
        self.unet_being_trained_index = -1 # keeps track of which unet is being trained at the moment
            
        for ind, one_unet in enumerate(unets):
            # check the class of the unet: accept Unet_OneD, NullUnet
            assert isinstance(one_unet, (Unet_OneD, Unet3D, NullUnet))
            is_first = ind == 0

            one_unet = one_unet.cast_model_parameters(
                lowres_cond = not is_first, # may open this channel
                cond_on_text = self.condition_on_text,
                text_embed_dim = self.text_embed_dim if self.condition_on_text else None,
                channels = self.channels,
                channels_out = self.channels,
            )
            
            if self.CKeys['Debug_Level']==Imagen_Init_Level:
                print (f"Add one UNet: ")
                print (one_unet)
                print (f"======================================== ")

            self.unets.append(one_unet)

        # determine whether we are training on images or video

        is_video = any([isinstance(unet, Unet3D) for unet in self.unets])
        self.is_video = is_video
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".is_video: {self.is_video}")

        self.right_pad_dims_to_datatype = partial(
            rearrange, 
            # --
            # pattern = ('b -> b 1 1 1' if not is_video else 'b -> b 1 1 1 1')
            # ++ may think adding one for video of 1d data
            pattern = ('b -> b 1 1' if not is_video else 'b -> b 1 1 1 1')
        )
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".right_pad_dims_to_datatype: {self.right_pad_dims_to_datatype}")

        self.resize_to = resize_video_to if is_video else resize_2d_image_to 
        # only triggered when the last dimension doesn't match the traget one
        # input: (mini-batch, channels, width) # assume it works for 1d
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        
        self.resize_to = partial(self.resize_to, mode = resize_mode)
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".resize_to: {self.resize_to}")

        # unet image sizes

        self.image_sizes = cast_tuple(image_sizes)
        assert num_unets == len(self.image_sizes), \
        f'you did not supply the correct number of u-nets ({len(self.unets)}) for resolutions {self.image_sizes}'
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".image_size: {self.image_sizes}")

        self.sample_channels = cast_tuple(self.channels, num_unets)
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".sample_channels: {self.sample_channels}")

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (num_unets - 1))), \
        'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'

        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.per_sample_random_aug_noise_level = per_sample_random_aug_noise_level
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".lowres_sample_noise_level: {self.lowres_sample_noise_level}")
            print (f".per_sample_random_aug_noise_level: {self.per_sample_random_aug_noise_level}")

        # classifier free guidance

        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".cond_drop_prob: {self.cond_drop_prob}")
            print (f".can_classifier_guidance: {self.can_classifier_guidance}")

        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity
        self.input_image_range = (0. if auto_normalize_img else -1., 1.)
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".normalize_img: {self.normalize_img}")
            print (f".unnormalize_img: {self.unnormalize_img}")
            print (f".input_image_range: {self.input_image_range}")

        # dynamic thresholding

        self.dynamic_thresholding = cast_tuple(dynamic_thresholding, num_unets)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".dynamic_thresholding: {self.dynamic_thresholding}")
            print (f".dynamic_thresholding_percentile: {self.dynamic_thresholding_percentile}")

        # temporal interpolations

        temporal_downsample_factor = cast_tuple(temporal_downsample_factor, num_unets)
        self.temporal_downsample_factor = temporal_downsample_factor

        self.resize_cond_video_frames = resize_cond_video_frames
        self.temporal_downsample_divisor = temporal_downsample_factor[0]
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".temporal_downsample_factor: {self.temporal_downsample_factor}")
            print (f".resize_cond_video_frames: {self.resize_cond_video_frames}")
            print (f".temporal_downsample_divisor: {self.temporal_downsample_divisor}")

        assert temporal_downsample_factor[-1] == 1, 'downsample factor of last stage must be 1'
        assert tuple(sorted(temporal_downsample_factor, reverse = True)) == temporal_downsample_factor, 'temporal downsample factor must be in order of descending'

        # elucidating parameters

        hparams = [
            num_sample_steps,
            sigma_min,
            sigma_max,
            sigma_data,
            rho,
            P_mean,
            P_std,
            S_churn,
            S_tmin,
            S_tmax,
            S_noise,
        ]

        hparams = [cast_tuple(hp, num_unets) for hp in hparams]
        self.hparams = [Hparams(*unet_hp) for unet_hp in zip(*hparams)]
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".hparams: {self.hparams}")

        # one temp parameter for keeping track of device

        self.register_buffer('_temp', torch.tensor([0.]), persistent = False)

        # default to device of unets passed in

        self.to(next(self.unets.parameters()).device)
        # ++
        if self.CKeys['Debug_Level']==Imagen_Init_Level:
            print (f".device: {next(self.unets.parameters()).device}")

    def force_unconditional_(self):
        self.condition_on_text = False
        self.unconditional = True

        for unet in self.unets:
            unet.cond_on_text = False

    @property
    def device(self):
        return self._temp.device

    def get_unet(self, unet_number):
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.unets]
            delattr(self, 'unets')
            self.unets = unets_list

        if index != self.unet_being_trained_index:
            for unet_index, unet in enumerate(self.unets):
                unet.to(self.device if unet_index == index else 'cpu')

        self.unet_being_trained_index = index
        return self.unets[index]

    def reset_unets_all_one_device(self, device = None):
        device = default(device, self.device)
        self.unets = nn.ModuleList([*self.unets])
        self.unets.to(device)

        self.unet_being_trained_index = -1

    @contextmanager
    def one_unet_in_gpu(self, unet_number = None, unet = None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        cpu = torch.device('cpu')

        devices = [module_device(unet) for unet in self.unets]

        self.unets.to(cpu)
        unet.to(self.device)

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    # overriding state dict functions

    def state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # dynamic thresholding

    def threshold_x_start(self, x_start, dynamic_threshold = True):
        if not dynamic_threshold:
            return x_start.clamp(-1., 1.)

        s = torch.quantile(
            rearrange(x_start, 'b ... -> b (...)').abs(),
            self.dynamic_thresholding_percentile,
            dim = -1
        )

        s.clamp_(min = 1.)
        s = right_pad_dims_to(x_start, s)
        return x_start.clamp(-s, s) / s

    # derived preconditioning params - Table 1

    def c_skip(self, sigma_data, sigma):
        return (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)

    def c_out(self, sigma_data, sigma):
        return sigma * sigma_data * (sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma_data, sigma):
        return 1 * (sigma ** 2 + sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

   # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(
        self,
        unet_forward,
        noised_images,
        sigma,
        *,
        sigma_data,
        clamp = False,
        dynamic_threshold = True,
        **kwargs
    ):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        padded_sigma = self.right_pad_dims_to_datatype(sigma)

        net_out = unet_forward(
            self.c_in(sigma_data, padded_sigma) * noised_images,
            self.c_noise(sigma),
            **kwargs
        )

        out = self.c_skip(sigma_data, padded_sigma) * noised_images \
        + self.c_out(sigma_data, padded_sigma) * net_out

        if not clamp:
            return out

        return self.threshold_x_start(out, dynamic_threshold)

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(
        self,
        num_sample_steps,
        rho,
        sigma_min,
        sigma_max
    ):
        N = num_sample_steps
        inv_rho = 1 / rho

        steps = torch.arange(
            num_sample_steps, 
            device = self.device, 
            dtype = torch.float32
        )
        sigmas = (
            sigma_max ** inv_rho \
            + steps / (N - 1) * (sigma_min ** inv_rho - sigma_max ** inv_rho)
        ) ** rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def one_unet_sample(
        self,
        unet,
        shape,
        *,
        unet_number,
        clamp = True,
        dynamic_threshold = True,
        cond_scale = 1.,
        use_tqdm = True,
        inpaint_videos = None,
        inpaint_images = None,
        inpaint_masks = None,
        inpaint_resample_times = 5,
        init_images = None,
        skip_steps = None,
        sigma_min = None,
        sigma_max = None,
        **kwargs
    ):
        # video

        is_video = len(shape) == 5
        frames = shape[-3] if is_video else None
        resize_kwargs = dict(target_frames = frames) if exists(frames) else dict()

        # get specific sampling hyperparameters for unet

        hp = self.hparams[unet_number - 1]

        sigma_min = default(sigma_min, hp.sigma_min)
        sigma_max = default(sigma_max, hp.sigma_max)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(
            hp.num_sample_steps, 
            hp.rho, 
            sigma_min, sigma_max
        )

        gammas = torch.where(
            (sigmas >= hp.S_tmin) & (sigmas <= hp.S_tmax),
            min(hp.S_churn / hp.num_sample_steps, sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # images is noise at the beginning

        init_sigma = sigmas[0]

        images = init_sigma * torch.randn(shape, device = self.device)

        # initializing with an image

        if exists(init_images):
            images += init_images

        # keeping track of x0, for self conditioning if needed

        x_start = None

        # prepare inpainting images and mask

        inpaint_images = default(inpaint_videos, inpaint_images)
        has_inpainting = exists(inpaint_images) and exists(inpaint_masks)
        resample_times = inpaint_resample_times if has_inpainting else 1

        if has_inpainting:
            inpaint_images = self.normalize_img(inpaint_images)
            inpaint_images = self.resize_to(inpaint_images, shape[-1], **resize_kwargs)
            inpaint_masks = self.resize_to(rearrange(inpaint_masks, 'b ... -> b 1 ...').float(), shape[-1], **resize_kwargs).bool()

        # unet kwargs

        unet_kwargs = dict(
            sigma_data = hp.sigma_data,
            clamp = clamp,
            dynamic_threshold = dynamic_threshold,
            cond_scale = cond_scale,
            **kwargs
        )

        # gradually denoise

        initial_step = default(skip_steps, 0)
        sigmas_and_gammas = sigmas_and_gammas[initial_step:]

        total_steps = len(sigmas_and_gammas)

        for ind, (sigma, sigma_next, gamma) in tqdm(
            enumerate(sigmas_and_gammas), 
            total = total_steps, 
            desc = 'sampling time step', 
            disable = not use_tqdm
        ):
            is_last_timestep = ind == (total_steps - 1)

            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            for r in reversed(range(resample_times)):
                is_last_resample_step = r == 0

                eps = hp.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

                sigma_hat = sigma + gamma * sigma
                added_noise = sqrt(sigma_hat ** 2 - sigma ** 2) * eps

                images_hat = images + added_noise

                self_cond = x_start if unet.self_cond else None

                if has_inpainting:
                    images_hat = images_hat * ~inpaint_masks + (inpaint_images + added_noise) * inpaint_masks

                model_output = self.preconditioned_network_forward(
                    unet.forward_with_cond_scale,
                    images_hat,
                    sigma_hat,
                    self_cond = self_cond,
                    **unet_kwargs
                )

                denoised_over_sigma = (images_hat - model_output) / sigma_hat

                images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

                # second order correction, if not the last timestep

                has_second_order_correction = sigma_next != 0

                if has_second_order_correction:
                    self_cond = model_output if unet.self_cond else None

                    model_output_next = self.preconditioned_network_forward(
                        unet.forward_with_cond_scale,
                        images_next,
                        sigma_next,
                        self_cond = self_cond,
                        **unet_kwargs
                    )

                    denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                    images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

                images = images_next

                if has_inpainting and not (is_last_resample_step or is_last_timestep):
                    # renoise in repaint and then resample
                    repaint_noise = torch.randn(shape, device = self.device)
                    images = images + (sigma - sigma_next) * repaint_noise

                x_start = model_output if not has_second_order_correction else model_output_next # save model output for self conditioning

        images = images.clamp(-1., 1.)

        if has_inpainting:
            images = images * ~inpaint_masks + inpaint_images * inpaint_masks

        return self.unnormalize_img(images)

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        # 1. on text condition
        texts: List[str] = None,
        text_masks = None,
        text_embeds = None,
        # 2. on condition images
        cond_images = None,
        cond_video_frames = None,
        post_cond_video_frames = None,
        # 3. inpaint images
        inpaint_videos = None,
        inpaint_images = None,
        inpaint_masks = None,
        inpaint_resample_times = 5,
        # 
        init_images = None,
        skip_steps = None,
        sigma_min = None,
        sigma_max = None,
        video_frames = None,
        batch_size = 1,
        cond_scale = 1.,
        lowres_sample_noise_level = None,
        start_at_unet_number = 1,
        start_image_or_video = None,
        stop_at_unet_number = None,
        return_all_unet_outputs = False,
        return_pil_images = False,
        use_tqdm = True,
        use_one_unet_in_gpu = True,
        device = None,
    ):
        # ++
        if self.CKeys['Debug_Level']==Imagen_Samp_Level:
            print (f"Debug mode for .sample func...")
            
        device = default(device, self.device)
        self.reset_unets_all_one_device(device = device)
        # ++
        if self.CKeys['Debug_Level']==Imagen_Samp_Level:
            print (f"device for unets: {device}")

        cond_images = maybe(cast_uint8_images_to_float)(cond_images)
        # ++
        if self.CKeys['Debug_Level']==Imagen_Samp_Level:
            if not cond_images==None:
                print (f"input cond_images.shape: {cond_images.shape}")
            else:
                print (f"input cond_images: None")
                
        # Channel t-1: use texts directly, not text_embeds; otherwise, text_embeds will be passed in
        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            assert all([*map(len, texts)]), 'text cannot be empty'

            with autocast(enabled = False):
                text_embeds, text_masks = self.encode_text(texts, return_attn_mask = True)

            text_embeds, text_masks = map(lambda t: t.to(device), (text_embeds, text_masks))
        
        if not self.unconditional:
            assert exists(text_embeds), 'text must be passed in if the network was not trained without text `condition_on_text` must be set to `False` when training'

            text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim = -1))
            batch_size = text_embeds.shape[0]
        # ++
        if self.CKeys['Debug_Level']==Imagen_Samp_Level:
            if not (text_embeds==None):
                print (f"text_embeds.shape: {text_embeds.shape}")
            if not (text_masks==None):
                print (f"text_masks.shape: {text_masks.shape}")

        # inpainting
        
        inpaint_images = default(inpaint_videos, inpaint_images)

        if exists(inpaint_images):
            if self.unconditional:
                if batch_size == 1: # assume researcher wants to broadcast along inpainted images
                    batch_size = inpaint_images.shape[0]

            assert inpaint_images.shape[0] == batch_size, 'number of inpainting images must be equal to the specified batch size on sample `sample(batch_size=<int>)``'
            assert not (self.condition_on_text and inpaint_images.shape[0] != text_embeds.shape[0]), 'number of inpainting images must be equal to the number of text to be conditioned on'
        
        assert not (self.condition_on_text and not exists(text_embeds)), 'text or text encodings must be passed into imagen if specified'
        assert not (not self.condition_on_text and exists(text_embeds)), 'imagen specified not to be conditioned on text, yet it is presented'
        assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        assert not (exists(inpaint_images) ^ exists(inpaint_masks)),  'inpaint images and masks must be both passed in to do inpainting'

        outputs = []

        is_cuda = next(self.parameters()).is_cuda
        device = next(self.parameters()).device
        
        # will be applied to the lowres images
        lowres_sample_noise_level = default(lowres_sample_noise_level, self.lowres_sample_noise_level)

        num_unets = len(self.unets)
        cond_scale = cast_tuple(cond_scale, num_unets)

        # handle video and frame dimension

        if self.is_video and exists(inpaint_images):
            video_frames = inpaint_images.shape[2]

            if inpaint_masks.ndim == 3:
                inpaint_masks = repeat(
                    inpaint_masks, 
                    # 'b h w -> b f h w', 
                    'b h -> b f h',
                    f = video_frames
                )

            assert inpaint_masks.shape[1] == video_frames

        assert not (self.is_video and not exists(video_frames)), 'video_frames must be passed in on sample time if training on video'

        # determine the frame dimensions, if needed

        all_frame_dims = calc_all_frame_dims(
            self.temporal_downsample_factor, 
            video_frames
        )

        # initializing with an image or video

        init_images = cast_tuple(init_images, num_unets)
        init_images = [maybe(self.normalize_img)(init_image) for init_image in init_images]
        # ++
        if self.CKeys['Debug_Level']==Imagen_Samp_Level:
            print (f"init_images: {init_images}")

        skip_steps = cast_tuple(skip_steps, num_unets)

        sigma_min = cast_tuple(sigma_min, num_unets)
        sigma_max = cast_tuple(sigma_max, num_unets)

        # handle starting at a unet greater than 1, for training only-upscaler training

        if start_at_unet_number > 1:
            assert start_at_unet_number <= num_unets, 'must start a unet that is less than the total number of unets'
            assert not exists(stop_at_unet_number) or start_at_unet_number <= stop_at_unet_number
            assert exists(start_image_or_video), 'starting image or video must be supplied if only doing upscaling'

            prev_image_size = self.image_sizes[start_at_unet_number - 2]
            img = self.resize_to(start_image_or_video, prev_image_size)

        # go through each unet in cascade

        for unet_number, unet, channel, image_size, frame_dims, unet_hparam, dynamic_threshold, unet_cond_scale, unet_init_images, unet_skip_steps, unet_sigma_min, unet_sigma_max in tqdm(
            zip(
                range(1, num_unets + 1), self.unets, 
                self.sample_channels, self.image_sizes, 
                all_frame_dims, self.hparams, 
                self.dynamic_thresholding, cond_scale, 
                init_images, skip_steps, 
                sigma_min, sigma_max
            ), 
            disable = not use_tqdm
        ):
            if unet_number < start_at_unet_number:
                continue

            assert not isinstance(unet, NullUnet), 'cannot sample from null unet'

            context = self.one_unet_in_gpu(unet = unet) if is_cuda and use_one_unet_in_gpu else nullcontext()

            with context:
                lowres_cond_img = lowres_noise_times = None
                
                # --
                # shape = (batch_size, channel, *frame_dims, image_size, image_size)
                # ++
                shape = (batch_size, channel, *frame_dims, image_size)

                resize_kwargs = dict()
                video_kwargs = dict()

                if self.is_video:
                    resize_kwargs = dict(target_frames = frame_dims[0])

                    video_kwargs = dict(
                        cond_video_frames = cond_video_frames,
                        post_cond_video_frames = post_cond_video_frames
                    )

                    video_kwargs = compact(video_kwargs)

                # handle video conditioning frames

                if self.is_video and self.resize_cond_video_frames:
                    downsample_scale = self.temporal_downsample_factor[unet_number - 1]
                    temporal_downsample_fn = partial(
                        scale_video_time, 
                        downsample_scale = downsample_scale
                    )
                    video_kwargs = maybe_transform_dict_key(
                        video_kwargs, 'cond_video_frames', 
                        temporal_downsample_fn
                    )
                    video_kwargs = maybe_transform_dict_key(
                        video_kwargs, 'post_cond_video_frames', 
                        temporal_downsample_fn
                    )

                # low resolution conditioning

                if unet.lowres_cond:
                    lowres_noise_times = self.lowres_noise_schedule.get_times(
                        batch_size, lowres_sample_noise_level, device = device
                    )

                    lowres_cond_img = self.resize_to(img, image_size, **resize_kwargs)
                    lowres_cond_img = self.normalize_img(lowres_cond_img)

                    lowres_cond_img, *_ = self.lowres_noise_schedule.q_sample(
                        x_start = lowres_cond_img, 
                        t = lowres_noise_times, 
                        noise = torch.randn_like(lowres_cond_img)
                    )

                if exists(unet_init_images):
                    unet_init_images = self.resize_to(
                        unet_init_images, image_size, **resize_kwargs
                    )

                # --
                # shape = (batch_size, self.channels, *frame_dims, image_size, image_size)
                # ++
                shape = (batch_size, self.channels, *frame_dims, image_size)

                img = self.one_unet_sample(
                    unet,
                    shape,
                    unet_number = unet_number,
                    text_embeds = text_embeds,
                    text_mask = text_masks,
                    cond_images = cond_images,
                    inpaint_images = inpaint_images,
                    inpaint_masks = inpaint_masks,
                    inpaint_resample_times = inpaint_resample_times,
                    init_images = unet_init_images,
                    skip_steps = unet_skip_steps,
                    sigma_min = unet_sigma_min,
                    sigma_max = unet_sigma_max,
                    cond_scale = unet_cond_scale,
                    lowres_cond_img = lowres_cond_img,
                    lowres_noise_times = lowres_noise_times,
                    dynamic_threshold = dynamic_threshold,
                    use_tqdm = use_tqdm,
                    **video_kwargs
                )

                outputs.append(img)

            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        output_index = -1 if not return_all_unet_outputs else slice(None) # either return last unet output or all unet outputs

        if not return_pil_images:
            return outputs[output_index]

        if not return_all_unet_outputs:
            outputs = outputs[-1:]

#         assert not self.is_video, 'automatically converting video tensor to video file for saving is not built yet'

#         pil_images = list(map(lambda img: list(map(T.ToPILImage(), img.unbind(dim = 0))), outputs))

#         return pil_images[output_index] # now you have a bunch of pillow images you can just .save(/where/ever/you/want.png)

# end of sampling ===================================================================================

    # training

    def loss_weight(self, sigma_data, sigma):
        return (sigma ** 2 + sigma_data ** 2) * (sigma * sigma_data) ** -2

    def noise_distribution(self, P_mean, P_std, batch_size):
        return (P_mean + P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(
        self,
        images, # rename to images or video
        unet: Union[Unet_OneD, Unet3D, NullUnet, DistributedDataParallel] = None,
        texts: List[str] = None,
        text_embeds = None,
        text_masks = None,
        unet_number = None,
        cond_images = None,
        **kwargs
    ):
        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            print (f"Now, in EImagen.forward() ...")
            
        if self.is_video and images.ndim == 4:
            # --
            # images = rearrange(images, 'b c h w -> b c 1 h w')
            # ++
            images = rearrange(images, 'b c h -> b c 1 h')
            kwargs.update(ignore_time = True)

        assert not (len(self.unets) > 1 and not exists(unet_number)), f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, 'you can only train on unet #{self.only_train_unet_number}'

        images = cast_uint8_images_to_float(images) # do nothing if input is not uint8: float btw (0, 1)
        cond_images = maybe(cast_uint8_images_to_float)(cond_images)
        # ++ for one_D need adjustment 
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            print (f"Reformat images and cond_images into float")
            print (f"images.shape: {images.shape}")
            print (f"images.dtype: {images.dtype}")
            print (f"max and min: {torch.max(images)} and {torch.min(images)}")
            if not (cond_images==None):
                print (f"cond_images.shape: {cond_images.shape}")
            else:
                print (f"cond_images: None")

        assert images.dtype == torch.float, f'images tensor needs to be floats but {images.dtype} dtype found instead'

        unet_index = unet_number - 1
        
        unet = default(unet, lambda: self.get_unet(unet_number))

        assert not isinstance(unet, NullUnet), 'null unet cannot and should not be trained'

        target_image_size    = self.image_sizes[unet_index]
        random_crop_size     = self.random_crop_sizes[unet_index]
        prev_image_size      = self.image_sizes[unet_index - 1] if unet_index > 0 else None
        hp                   = self.hparams[unet_index]
        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            print (f"target_image_size: {target_image_size}")
            print (f"random_crop_size: {random_crop_size}")
            print (f"prev_image_size: {prev_image_size}")
            print (f"hp: {hp}")
        
        # --
        # batch_size, c, *_, h, w, device, is_video = *images.shape, images.device, (images.ndim == 5)
        # ++
        batch_size, c, *_, h, device, is_video = *images.shape, images.device, (images.ndim == 4)
        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            print (f"batch_size: {batch_size}")
            print (f"channel c: {c}")
            print (f"1d image size, h: {h} ")
            

        frames              = images.shape[2] if is_video else None
        all_frame_dims      = tuple(
            safe_get_tuple_index(el, 0) for el in calc_all_frame_dims(
                self.temporal_downsample_factor, frames
            )
        )
        ignore_time         = kwargs.get('ignore_time', False)
        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            print (f"frames: {frames}")
            print (f"all_frame_dims: {all_frame_dims}")
            print (f"ignore_time: {ignore_time}")

        target_frame_size   = all_frame_dims[unet_index] if is_video and not ignore_time else None
        prev_frame_size     = all_frame_dims[unet_index - 1] if is_video and not ignore_time and unet_index > 0 else None
        frames_to_resize_kwargs = lambda frames: dict(target_frames = frames) if exists(frames) else dict()
        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            print (f"target_frame_size: {target_frame_size}")
            print (f"prev_frame_size: {prev_frame_size}")
            print (f"frames_to_resize_kwargs: {frames_to_resize_kwargs}")

        assert images.shape[1] == self.channels
        assert h >= target_image_size # and w >= target_image_size
        
        # texts provided, not text_embeds
        # 
        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            assert all([*map(len, texts)]), 'text cannot be empty'
            assert len(texts) == len(images), 'number of text captions does not match up with the number of images given'

            with autocast(enabled = False):
                text_embeds, text_masks = self.encode_text(texts, return_attn_mask = True)

            text_embeds, text_masks = map(lambda t: t.to(images.device), (text_embeds, text_masks))
        # now we have text_embeds, and text_masks

        if not self.unconditional:
            text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim = -1))
        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            # print (f"text_masks: \n{text_masks}")
            print (f"text_masks.shape: \n{text_masks.shape}")

        assert not (
            self.condition_on_text and not exists(text_embeds)
        ), 'text or text encodings must be passed into decoder if specified'
        assert not (
            not self.condition_on_text and exists(text_embeds)
        ), 'decoder specified not to be conditioned on text, yet it is presented'

        assert not (
            exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim
        ), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        # handle video conditioning frames

        if self.is_video and self.resize_cond_video_frames:
            downsample_scale = self.temporal_downsample_factor[unet_index]
            temporal_downsample_fn = partial(scale_video_time, downsample_scale = downsample_scale)
            kwargs = maybe_transform_dict_key(kwargs, 'cond_video_frames', temporal_downsample_fn)
            kwargs = maybe_transform_dict_key(kwargs, 'post_cond_video_frames', temporal_downsample_fn)

        # low resolution conditioning
        # this part is on if the trained one is the 2nd unet

        lowres_cond_img = lowres_aug_times = None
        if exists(prev_image_size): # so, this is the 2nd unet
            # ++
            if self.CKeys['Debug_Level']==Imagen_Forw_Level:
                print (f"prev_image_size detected. So, this is for 2nd UNet")
                print (f"Create lowres_cond_img by resizing true image")
                print (f"    images.shape: {images.shape}")
            lowres_cond_img = self.resize_to(
                images, 
                prev_image_size, 
                **frames_to_resize_kwargs(prev_frame_size), 
                clamp_range = self.input_image_range
            )
            # ++
            if self.CKeys['Debug_Level']==Imagen_Forw_Level:
                print (f"    1. resize full image to previous size: coarsening")
                print (f"    .resize_to(images,prev_image_size)->lowres_cond_img.shape: {lowres_cond_img.shape}")
            lowres_cond_img = self.resize_to(
                lowres_cond_img, target_image_size, 
                **frames_to_resize_kwargs(target_frame_size), 
                clamp_range = self.input_image_range
            )
            # ++
            if self.CKeys['Debug_Level']==Imagen_Forw_Level:
                print (f"    2. fit into the traget size: only change size")
                print (f"    .resize_to(lowres_cond_img,target_image_size)->lowres_cond_img.shape: {lowres_cond_img.shape}")

            if self.per_sample_random_aug_noise_level:
                lowres_aug_times = self.lowres_noise_schedule.sample_random_times(
                    batch_size, device = device
                )
            else: # i.e., all samples in the batch use the same
                lowres_aug_time = self.lowres_noise_schedule.sample_random_times(
                    1, device = device
                )
                lowres_aug_times = repeat(lowres_aug_time, '1 -> b', b = batch_size)
            # ++
            if self.CKeys['Debug_Level']==Imagen_Forw_Level:
                print (f"get noise schedule for lowres_cond_img, lowres_aug_time.shape: {lowres_aug_times.shape}")

        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            print (f"images.shape: {images.shape}")
        images = self.resize_to(
            images, 
            target_image_size, 
            **frames_to_resize_kwargs(target_frame_size)
        )
        # not triggered if images.shape[-1]==target_image_size
        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            print (f".resize_to() -> images.shape: {images.shape}")

        # normalize to [-1, 1]
        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            print (f"Bef. normalize_img, min/ax of images: {torch.max(images)}, {torch.min(images)}")
            print (f".normalize_img: {self.normalize_img}")
        images = self.normalize_img(images) # assume images (0,1)->(-1,1)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)
        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            print (f"After normalize_img, should be [-1,1]")
            print (f"images max and min: {torch.max(images)} and {torch.min(images)}")
            if exists(lowres_cond_img):
                print (f"lowres_cond_img max and min: {torch.max(lowres_cond_img)} and {torch.min(lowres_cond_img)}")

        # random cropping during training
        # for upsamplers

        if exists(random_crop_size):
            aug = K.RandomCrop((random_crop_size, random_crop_size), p = 1.)

            if is_video:
                images, lowres_cond_img = map(
                    # --
                    # lambda t: rearrange(t, 'b c f h w -> (b f) c h w'),
                    # ++
                    lambda t: rearrange(t, 'b c f h -> (b f) c h'), 
                    (images, lowres_cond_img)
                )

            # make sure low res conditioner and image both get augmented the same way
            # detailed https://kornia.readthedocs.io/en/latest/augmentation.module.html?highlight=randomcrop#kornia.augmentation.RandomCrop
            images = aug(images)
            lowres_cond_img = aug(lowres_cond_img, params = aug._params)

            if is_video:
                images, lowres_cond_img = map(
                    # --
                    # lambda t: rearrange(t, '(b f) c h w -> b c f h w', f = frames), 
                    # ++
                    lambda t: rearrange(t, '(b f) c h -> b c f h', f = frames), 
                    (images, lowres_cond_img)
                )

        # noise the lowres conditioning image
        # at sample time, they then fix the noise level of 0.1 - 0.3

        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_cond_img_noisy, *_ = self.lowres_noise_schedule.q_sample(
                x_start = lowres_cond_img, 
                t = lowres_aug_times, 
                noise = torch.randn_like(lowres_cond_img)
            )
            # ++
            if self.CKeys['Debug_Level']==Imagen_Forw_Level:
                print (f"add noise to lowres_cond_img...")
                print (f"lowres_cond_img_noisy.shape: {lowres_cond_img_noisy.shape}")
            

        # get the sigmas

        sigmas = self.noise_distribution(
            hp.P_mean, hp.P_std, batch_size
        )
        padded_sigmas = self.right_pad_dims_to_datatype(sigmas)
        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            print (f"sigmas.shape: {sigmas.shape}")
            print (f"padded_sigmas.shape: {padded_sigmas.shape}")

        # noise

        noise = torch.randn_like(images)
        noised_images = images + padded_sigmas * noise  # alphas are 1. in the paper
        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            print (f"add noise into images")

        # unet kwargs

        unet_kwargs = dict(
            sigma_data = hp.sigma_data,
            text_embeds = text_embeds,
            text_mask = text_masks,
            cond_images = cond_images,
            lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_aug_times),
            lowres_cond_img = lowres_cond_img_noisy,
            cond_drop_prob = self.cond_drop_prob,
            **kwargs
        )

        # self conditioning - https://arxiv.org/abs/2208.04202 - training will be 25% slower

        # Because 'unet' can be an instance of DistributedDataParallel coming from the
        # ImagenTrainer.unet_being_trained when invoking ImagenTrainer.forward(), we need to
        # access the member 'module' of the wrapped unet instance.
        self_cond = unet.module.self_cond if isinstance(unet, DistributedDataParallel) else unet.self_cond

        if self_cond and random() < 0.5:
            # ++
            if self.CKeys['Debug_Level']==Imagen_Forw_Level:
                print (f"self_cond is triggered.")
                print (f"get into unet.......")
            
            with torch.no_grad():
                pred_x0 = self.preconditioned_network_forward(
                    unet.forward,
                    noised_images,
                    sigmas,
                    **unet_kwargs
                ).detach()
            # ++
            if self.CKeys['Debug_Level']==Imagen_Forw_Level:
                print (f"get out of unet.......")
                print (f"prop noised_images via the net.")
                print (f"get pred_x0.shape: {pred_x0.shape}")

            unet_kwargs = {**unet_kwargs, 'self_cond': pred_x0}

        # get prediction
        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            # print (f"unet_kwargs: \n{unet_kwargs}")
            print (f"unet_kwargs include keys: ")
            for this_key in unet_kwargs.keys():
                print ("    "+this_key)
                if torch.is_tensor(unet_kwargs[this_key]):
                    print (f"    {unet_kwargs[this_key].shape}")
                else:
                    print (f"    {type(unet_kwargs[this_key])}")

        denoised_images = self.preconditioned_network_forward(
            unet.forward,
            noised_images,
            sigmas,
            **unet_kwargs
        )
        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            print (f"Back to EImagen, get denoised_images")
            print (f"denoised_images.shape: {denoised_images.shape}")

        # losses

        losses = F.mse_loss(denoised_images, images, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        # loss weighting

        losses = losses * self.loss_weight(hp.sigma_data, sigmas)
        # ++
        if self.CKeys['Debug_Level']==Imagen_Forw_Level:
            print (f"postprocess losses based on hp.sigma_data")

        # return average loss

        return losses.mean()
