"""
Task:
1. create a trainer for ProteinDesigner
2. include train_loop, sample_loop

Bo Ni, Sep 8, 2024
"""

# //////////////////////////////////////////////////////
# 0. load in packages
# //////////////////////////////////////////////////////

import os
from math import ceil
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler

import pytorch_warmup as warmup

from packaging import version

import numpy as np

from ema_pytorch import EMA

from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem

# //////////////////////////////////////////////////////////////
# 2. special packages
# //////////////////////////////////////////////////////////////
from VibeGen.ModelPack import (
    ProteinDesigner_Base
)
from VibeGen.imagen_x_imagen_pytorch import (
    ElucidatedImagen_OneD, eval_decorator
)

# //////////////////////////////////////////////////////////////
# 3. local setup parameters: for debug purpose
# //////////////////////////////////////////////////////////////
PT_Init_Level = 1
PT_Forw_Level = 1

# //////////////////////////////////////////////////////////////
# 4. helper functions
# //////////////////////////////////////////////////////////////
def cycle(dl):
    while True:
        for data in dl:
            yield data
            
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(val, length = 1):
    if isinstance(val, list):
        val = tuple(val)

    return val if isinstance(val, tuple) else ((val,) * length)

def find_first(fn, arr):
    for ind, el in enumerate(arr):
        if fn(el):
            return ind
    return -1

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# url to fs, bucket, path - for checkpointing to cloud

def url_to_bucket(url):
    if '://' not in url:
        return url

    _, suffix = url.split('://')

    if prefix in {'gs', 's3'}:
        return suffix.split('/')[0]
    else:
        raise ValueError(f'storage type prefix "{prefix}" is not supported yet')

# decorators

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def cast_torch_tensor(fn, cast_fp16 = False):
    @wraps(fn)
    def inner(model, *args, **kwargs):
        device = kwargs.pop('_device', model.device)
        cast_device = kwargs.pop('_cast_device', True)

        should_cast_fp16 = cast_fp16 and model.cast_half_at_training

        kwargs_keys = kwargs.keys()
        all_args = (*args, *kwargs.values())
        split_kwargs_index = len(all_args) - len(kwargs_keys)
        all_args = tuple(map(lambda t: torch.from_numpy(t) if exists(t) and isinstance(t, np.ndarray) else t, all_args))

        if cast_device:
            all_args = tuple(map(lambda t: t.to(device) if exists(t) and isinstance(t, torch.Tensor) else t, all_args))

        if should_cast_fp16:
            all_args = tuple(map(lambda t: t.half() if exists(t) and isinstance(t, torch.Tensor) and t.dtype != torch.bool else t, all_args))

        args, kwargs_values = all_args[:split_kwargs_index], all_args[split_kwargs_index:]
        kwargs = dict(tuple(zip(kwargs_keys, kwargs_values)))

        out = fn(model, *args, **kwargs)
        return out
    return inner

# gradient accumulation functions

def split_iterable(it, split_size):
    accum = []
    for ind in range(ceil(len(it) / split_size)):
        start_index = ind * split_size
        accum.append(it[start_index: (start_index + split_size)])
    return accum

def split(t, split_size = None):
    if not exists(split_size):
        return t

    if isinstance(t, torch.Tensor):
        return t.split(split_size, dim = 0)

    if isinstance(t, Iterable):
        return split_iterable(t, split_size)

    return TypeError

def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None

def split_args_and_kwargs(*args, split_size = None, **kwargs):
    all_args = (*args, *kwargs.values())
    len_all_args = len(all_args)
    first_tensor = find_first(lambda t: isinstance(t, torch.Tensor), all_args)
    assert exists(first_tensor)

    batch_size = len(first_tensor)
    split_size = default(split_size, batch_size)
    num_chunks = ceil(batch_size / split_size)

    dict_len = len(kwargs)
    dict_keys = kwargs.keys()
    split_kwargs_index = len_all_args - dict_len

    split_all_args = [split(arg, split_size = split_size) if exists(arg) and isinstance(arg, (torch.Tensor, Iterable)) else ((arg,) * num_chunks) for arg in all_args]
    chunk_sizes = num_to_groups(batch_size, split_size)

    for (chunk_size, *chunked_all_args) in tuple(zip(chunk_sizes, *split_all_args)):
        chunked_args, chunked_kwargs_values = chunked_all_args[:split_kwargs_index], chunked_all_args[split_kwargs_index:]
        chunked_kwargs = dict(tuple(zip(dict_keys, chunked_kwargs_values)))
        chunk_size_frac = chunk_size / batch_size
        yield chunk_size_frac, (chunked_args, chunked_kwargs)


# imagen trainer

def imagen_sample_in_chunks(fn):
    @wraps(fn)
    def inner(self, *args, max_batch_size = None, **kwargs):
        if not exists(max_batch_size):
            return fn(self, *args, **kwargs)

        if self.imagen.unconditional:
            batch_size = kwargs.get('batch_size')
            batch_sizes = num_to_groups(batch_size, max_batch_size)
            outputs = [fn(self, *args, **{**kwargs, 'batch_size': sub_batch_size}) for sub_batch_size in batch_sizes]
        else:
            outputs = [fn(self, *chunked_args, **chunked_kwargs) for _, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs)]

        if isinstance(outputs[0], torch.Tensor):
            return torch.cat(outputs, dim = 0)

        return list(map(lambda t: torch.cat(t, dim = 0), list(zip(*outputs))))

    return inner


def restore_parts(state_dict_target, state_dict_from):
    for name, param in state_dict_from.items():

        if name not in state_dict_target:
            continue

        if param.size() == state_dict_target[name].size():
            state_dict_target[name].copy_(param)
        else:
            print(f"layer {name}({param.size()} different than target: {state_dict_target[name].size()}")

    return state_dict_target

# //////////////////////////////////////////////////////////////
# 5. Main class:
# //////////////////////////////////////////////////////////////

class ProteinDesigner_Trainer(nn.Module):
    locked = False
    
    def __init__(
        self,
        # 1. on models
        ProtDesi = None,                  # provide a object
        ProtDesi_checkpoint_path = None,  # provide a checkpoint path
        only_train_unet_number = None,
        # 2. on optimizer
        use_ema = True,
        lr = 1e-4,
        eps = 1e-8,
        beta1 = 0.9,
        beta2 = 0.99,
        max_grad_norm = None,
        group_wd_params = True,
        warmup_steps = None,
        cosine_decay_max_steps = None,
        
        fp16 = False,
        precision = None,
        split_batches = True,
        dl_tuple_output_keywords_names = ('images', 'text_embeds', 'text_masks', 'cond_images'),
        verbose = True,
        split_valid_fraction = 0.025,
        split_valid_from_train = False,
        split_random_seed = 42,
        checkpoint_path = None,
        checkpoint_every = None,
        checkpoint_fs = None,
        fs_kwargs: dict = None,
        max_checkpoints_keep = 20,
        # ++
        CKeys = {'Debug_Level':0},
        **kwargs
    ):
        super().__init__()
        
        # 0. asserts some
        # .....................................................
        assert not ProteinDesigner_Trainer.locked, 'ProteinDesigner_Trainer can only be initialized once per process - for the sake of distributed training, you will now have to create a separate script to train each unet (or a script that accepts unet number as an argument)'
        
        assert exists(ProtDesi) ^ exists(ProtDesi_checkpoint_path), 'either imagen instance is passed into the trainer, or a checkpoint path that contains the imagen config'
        
        # ++
        self.CKeys = CKeys
        if self.CKeys['Debug_Level']==PT_Init_Level:
            print (f"|||||||||||||||||||||||||||||||||||||||||||||||||||")
            print (f"Initialize Protein_Designer Trainer object...")
            
        # determine filesystem, using fsspec, for saving to local filesystem or cloud
        self.fs = checkpoint_fs
        
        if not exists(self.fs):
            fs_kwargs = default(fs_kwargs, {})
            self.fs, _ = url_to_fs(
                default(checkpoint_path, './'), **fs_kwargs
            )
        # ++ 
        if self.CKeys['Debug_Level']==PT_Init_Level:
            print (f"file system: .fs: {self.fs}")
            
        assert isinstance(ProtDesi, (ProteinDesigner_Base)), \
        "ProtDesi is not from ProteinDesigner_Base"
        
        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)
        # ++ 
        if self.CKeys['Debug_Level']==PT_Init_Level:
            print (f"ema_kwargs: {ema_kwargs}")
            print (f"kwargs: {kwargs}")
            
        self.is_elucidated = isinstance(
            ProtDesi.diffuser_core, ElucidatedImagen_OneD
        )
        
        # create accelerator instance
        
        accelerate_kwargs, kwargs = groupby_prefix_and_trim(
            'accelerate_', kwargs
        )
        # ++ 
        if self.CKeys['Debug_Level']==PT_Init_Level:
            print (f"create acce instance...")
            print (f"accelerate_kwargs: {accelerate_kwargs}")
            print (f"kwargs: {kwargs}")
        
        assert not (fp16 and exists(precision)), \
        'either set fp16 = True or forward the precision ("fp16", "bf16") to Accelerator'
        accelerator_mixed_precision = default(
            precision, 
            'fp16' if fp16 else 'no'
        )
        
        self.accelerator = Accelerator(**{
            'split_batches': split_batches,
            'mixed_precision': accelerator_mixed_precision,
            'kwargs_handlers': [
                DistributedDataParallelKwargs(find_unused_parameters = True)
            ], 
            **accelerate_kwargs})
        
        # .is_distributed is a self fun
        ProteinDesigner_Trainer.locked = self.is_distributed
        # ++ 
        if self.CKeys['Debug_Level']==PT_Init_Level:
            print (f".is_distributed or .locked: {ProteinDesigner_Trainer.locked}")
            
        # cast data to fp16 at training time if needed
        self.cast_half_at_training = accelerator_mixed_precision == 'fp16'
        
        # grad scaler must be managed outside of accelerator
        grad_scaler_enabled = fp16
        
        # ProteinDesigner, imagen, unets and ema unets
        self.ProtDesi = ProtDesi
        self.imagen = ProtDesi.diffuser_core # imagen
        self.num_unets = len(self.imagen.unets)
        
        self.use_ema = use_ema and self.is_main
        self.ema_unets = nn.ModuleList([])
        # ++ 
        if self.CKeys['Debug_Level']==PT_Init_Level:
            print (f".num_unets: {self.num_unets}")
            print (f".use_ema: {self.use_ema}")
            
        # keep track of what unet is being trained on
        # only going to allow 1 unet training at a time
        
        self.ema_unet_being_trained_index = -1 
        # keeps track of which ema unet is being trained on

        # data related functions
        
        self.train_dl_iter = None
        self.train_dl = None
        
        self.valid_dl_iter = None
        self.valid_dl = None
        
        self.dl_tuple_output_keywords_names = dl_tuple_output_keywords_names
        
        # auto splitting validation from training, if dataset is passed in
        
        self.split_valid_from_train = split_valid_from_train
        
        assert 0 <= split_valid_fraction <= 1, \
        'split valid fraction must be between 0 and 1'
        self.split_valid_fraction = split_valid_fraction
        self.split_random_seed = split_random_seed
        
        # be able to finely customize learning rate, weight decay
        # per unet
        
        # ++
        if self.CKeys['Debug_Level']==PT_Init_Level:
            print (f" Finely customize learning rate, weight decay")
                
        lr, eps, warmup_steps, cosine_decay_max_steps = map(
            partial(cast_tuple, length = self.num_unets), 
            (lr, eps, warmup_steps, cosine_decay_max_steps)
        )
        
        for ind, (
            unet, unet_lr, unet_eps, 
            unet_warmup_steps, unet_cosine_decay_max_steps
        ) in enumerate(
            zip(
                self.imagen.unets, 
                lr, eps, warmup_steps, cosine_decay_max_steps
            )
        ):

            optimizer = Adam(
                unet.parameters(),
                lr = unet_lr,
                eps = unet_eps,
                betas = (beta1, beta2),
                **kwargs
            )

            if self.use_ema:
                self.ema_unets.append(EMA(unet, **ema_kwargs))

            scaler = GradScaler(enabled = grad_scaler_enabled)

            scheduler = warmup_scheduler = None

            if exists(unet_cosine_decay_max_steps):
                scheduler = CosineAnnealingLR(
                    optimizer, 
                    T_max = unet_cosine_decay_max_steps
                )

            if exists(unet_warmup_steps):
                warmup_scheduler = warmup.LinearWarmup(
                    optimizer, 
                    warmup_period = unet_warmup_steps
                )

                if not exists(scheduler):
                    scheduler = LambdaLR(
                        optimizer, 
                        lr_lambda = lambda step: 1.0
                    )

            # set on object

            setattr(self, f'optim{ind}', optimizer) # cannot use pytorch ModuleList for some reason with optimizers
            setattr(self, f'scaler{ind}', scaler)
            setattr(self, f'scheduler{ind}', scheduler)
            setattr(self, f'warmup{ind}', warmup_scheduler)
            
            # ++
            if self.CKeys['Debug_Level']==PT_Init_Level:
                print (f"    on Unit-{ind}")
                print (f"    scaler: {scaler}")
                print (f"    scheduler: {scheduler}")
                print (f"    warmup_scheduler: {warmup_scheduler}")
        

        # gradient clipping if needed
        
        self.max_grad_norm = max_grad_norm
        
        # step tracker and misc
        
        self.register_buffer('steps', torch.tensor([0] * self.num_unets))
        
        self.verbose = verbose
        
        # automatic set devices based on what accelerator decided
        
        # self.imagen.to(self.device)
        self.ProtDesi.to(self.device)
        self.to(self.device)
        
        # checkpointing
        
        assert not (exists(checkpoint_path) ^ exists(checkpoint_every))
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.max_checkpoints_keep = max_checkpoints_keep
        
        self.can_checkpoint = self.is_local_main \
        if isinstance(checkpoint_fs, LocalFileSystem) else self.is_main
        
        # ++
        if self.CKeys['Debug_Level']==PT_Init_Level:
            print (f".checkpoint_path: {self.checkpoint_path}")
            print (f".checkpoint_every: {self.checkpoint_every}")
            print (f".max_checkpoints_keep: {self.max_checkpoints_keep}")
            print (f".can_checkpoint: {self.can_checkpoint}")
        
        if exists(checkpoint_path) and self.can_checkpoint:
            bucket = url_to_bucket(checkpoint_path)

            if not self.fs.exists(bucket):
                self.fs.mkdir(bucket)

            self.load_from_checkpoint_folder()
            
        # only allowing training for unet

        self.only_train_unet_number = only_train_unet_number
        self.prepared = False
        # ++
        if self.CKeys['Debug_Level']==PT_Init_Level:
            print (f".only_train_unet_number: {self.only_train_unet_number}")
            print (f".prepared: {self.prepared}")

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# computed values
    @property
    def device(self):
        return self.accelerator.device
    
    @property
    def is_distributed(self):
        return not (
            self.accelerator.distributed_type == DistributedType.NO \
            and self.accelerator.num_processes == 1
        )
    
    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    @property
    def unwrapped_unet(self):
        return self.accelerator.unwrap_model(self.unet_being_trained)
    
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# optimizer helper functions


# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

    def load_from_checkpoint_folder(
        self, 
        last_total_steps = -1
    ):
        if last_total_steps != -1:
            filepath = os.path.join(
                self.checkpoint_path, 
                f'checkpoint.{last_total_steps}.pt'
            )
            self.load(filepath)
            return

        sorted_checkpoints = self.all_checkpoints_sorted

        if len(sorted_checkpoints) == 0:
            self.print(
                f'no checkpoints found to load from at {self.checkpoint_path}'
            )
            return

        last_checkpoint = sorted_checkpoints[0]
        self.load(last_checkpoint)
    





    # ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # Forward_Pack
    
    # validating the unet number

    def validate_unet_number(
        self, 
        unet_number = None
    ):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        assert 0 < unet_number <= self.num_unets, \
        f'unet number should be in between 1 and {self.num_unets}'
        
        return unet_number
    
    # function for allowing only one unet from being trained at a time

    def validate_and_set_unet_being_trained(self, unet_number = None):
        if exists(unet_number):
            self.validate_unet_number(unet_number)

        assert not exists(self.only_train_unet_number) or \
        self.only_train_unet_number == unet_number, \
        'you can only train on one unet at a time. you will need to save the trainer into a checkpoint, and resume training on a new unet'

        self.only_train_unet_number = unet_number
        self.imagen.only_train_unet_number = unet_number

        if not exists(unet_number):
            return

        self.wrap_unet(unet_number)
    
    
    def wrap_unet(self, unet_number):
        if hasattr(self, 'one_unet_wrapped'):
            return

        unet = self.imagen.get_unet(unet_number)
        unet_index = unet_number - 1

        optimizer = getattr(self, f'optim{unet_index}')
        scheduler = getattr(self, f'scheduler{unet_index}')

        if self.train_dl:
            self.unet_being_trained, self.train_dl, optimizer\
            = self.accelerator.prepare(
                unet, self.train_dl, optimizer
            )
        else:
            self.unet_being_trained, optimizer\
            = self.accelerator.prepare(unet, optimizer)

        if exists(scheduler):
            scheduler = self.accelerator.prepare(scheduler)

        setattr(self, f'optim{unet_index}', optimizer)
        setattr(self, f'scheduler{unet_index}', scheduler)

        self.one_unet_wrapped = True
        
    # hacking accelerator due to not having separate gradscaler per optimizer

    def set_accelerator_scaler(self, unet_number):
        
        def patch_optimizer_step(accelerated_optimizer, method):
            def patched_step(*args, **kwargs):
                accelerated_optimizer._accelerate_step_called = True
                return method(*args, **kwargs)
            return patched_step

        unet_number = self.validate_unet_number(unet_number)
        scaler = getattr(self, f'scaler{unet_number - 1}')

        self.accelerator.scaler = scaler
        for optimizer in self.accelerator._optimizers:
            optimizer.scaler = scaler
            optimizer._accelerate_step_called = False
            optimizer._optimizer_original_step_method = optimizer.optimizer.step
            optimizer._optimizer_patched_step_method = patch_optimizer_step(
                optimizer, optimizer.optimizer.step
            )
            
        
    
    @partial(cast_torch_tensor, cast_fp16 = True)
    def forward(
        self,
        *args,
        unet_number = None,
        max_batch_size = None,
        **kwargs
    ):
        # ++
        if self.CKeys['Debug_Level']==PT_Forw_Level:
            print (f"Debug mode for trainer.forward...")
            
        unet_number = self.validate_unet_number(unet_number) # check if unet_number is in the range
        # ++
        if self.CKeys['Debug_Level']==PT_Forw_Level:
            print (f"Train UNet number: {unet_number}")
            
        self.validate_and_set_unet_being_trained(unet_number)
        self.set_accelerator_scaler(unet_number)

        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, f'you can only train unet #{self.only_train_unet_number}'

        total_loss = 0.

        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs):
            
            with self.accelerator.autocast():
                #--
                # loss = self.imagen(
                #++
                loss = self.ProtDesi(
                    *chunked_args, 
                    unet = self.unet_being_trained, 
                    unet_number = unet_number, **chunked_kwargs
                )
                loss = loss * chunk_size_frac
            # ++
            if self.CKeys['Debug_Level']==PT_Forw_Level:
                print (f"get loss for a fraction: {loss}")
                
            total_loss += loss.item()
            # ++
            if self.CKeys['Debug_Level']==PT_Forw_Level:
                print (f"update tot_loss: {total_loss}")

            if self.training:
                self.accelerator.backward(loss)

        return total_loss
