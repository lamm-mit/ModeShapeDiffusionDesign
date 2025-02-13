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
import math

from ema_pytorch import EMA

from einops import rearrange

from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem

import shutil
import matplotlib.pyplot as plt

# //////////////////////////////////////////////////////////////
# 2. special packages
# //////////////////////////////////////////////////////////////
from VibeGen.DataSetPack import (
    pad_a_np_arr_esm_for_NMS
)
from VibeGen.ModelPack import (
    ProteinDesigner_Base,
    ProteinPredictor_Base
)
from VibeGen.imagen_x_imagen_pytorch import (
    ElucidatedImagen_OneD, eval_decorator
)
#
from VibeGen.UtilityPack import (
    Print, Print_model_params,
    get_toks_list_from_Y_batch,
    save_2d_tensor_as_np_arr_txt,
    write_fasta_file,
    compare_two_seq_strings,
    fold_one_AA_to_SS_using_omegafold,
    show_pdb,
    get_DSSP_result,
    write_DSSP_result_to_json,
    write_one_line_to_file,
    decode_many_ems_token_rec,
    get_nms_vec_as_arr_list_from_batch_using_mask,
    compare_two_nms_vecs_arr
)

# //////////////////////////////////////////////////////////////
# 3. local setup parameters: for debug purpose
# //////////////////////////////////////////////////////////////
PT_Init_Level = 1
PT_Forw_Level = 1

Local_Debug_Level = 0
# //////////////////////////////////////////////////////////////
# 4. helper functions
# //////////////////////////////////////////////////////////////

# //////////////////////////////////////////////////////////////
# 5. Main class/functions: a base trainer wrap for ProteinDesigner
# //////////////////////////////////////////////////////////////
        
        
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# helpers for training

def get_trainable_param_dict(
    model,
    key_name_prefix="",
):  
    param_dict={}
    for pn, p in model.named_parameters():
        if p.requires_grad:
            full_key_name = key_name_prefix+pn
            param_dict[full_key_name] = p
        
    return param_dict

# ++ for PP
def get_grouped_params_using_dimension_from_PP(
    model,
    weight_decay=0.0 # no decay as default
):
    assert isinstance(model, (ProteinPredictor_Base)), \
    "Only ProteinPredictor_Base is allowed for this fun."
    
    # NOTE: model.named_parameters() misses unet part once 
    # the model is called. Here, we double check unet part
    
    # 1. get the full list as two dict
    param_dict_top = get_trainable_param_dict(model)
    param_dict_unet = get_trainable_param_dict(
        model.diffuser_core.unets[0],
        key_name_prefix=f"diffuser_core.unets.0."
    )
    
    # 2. merge without repeating
    # NOTE: the method 
    # dict1 = {'a': 1, 'b': 2}
    # dict2 = {'b': 3, 'c': 4}
    # print (dict1 | dict2)
    # > {'a': 1, 'b': 3, 'c': 4}
    # so, dict2 will overwrite dict1
    
    param_dict = param_dict_unet | param_dict_top
    
    # 3. may add finer control of the parameter groups
    # 
    # create optim groups. 
    # Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, 
    # all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    # 
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    Print (f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    Print (f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    Print (f"In tot: {num_nodecay_params+num_decay_params}")

    return optim_groups

def get_grouped_params_using_dimension_from_PD(
    model,
    weight_decay=0.0 # no decay as default
):
    assert isinstance(model, (ProteinDesigner_Base)), \
    "Only ProteinDesigner_Base is allowed for this fun."
    
    # NOTE: model.named_parameters() misses unet part once 
    # the model is called. Here, we double check unet part
    
    # 1. get the full list as two dict
    param_dict_top = get_trainable_param_dict(model)
    param_dict_unet = get_trainable_param_dict(
        model.diffuser_core.unets[0],
        key_name_prefix=f"diffuser_core.unets.0."
    )
    
    # 2. merge without repeating
    # NOTE: the method 
    # dict1 = {'a': 1, 'b': 2}
    # dict2 = {'b': 3, 'c': 4}
    # print (dict1 | dict2)
    # > {'a': 1, 'b': 3, 'c': 4}
    # so, dict2 will overwrite dict1
    
    param_dict = param_dict_unet | param_dict_top
    
    # 3. may add finer control of the parameter groups
    # 
    # create optim groups. 
    # Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, 
    # all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    # 
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    Print (f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    Print (f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    Print (f"In tot: {num_nodecay_params+num_decay_params}")

    return optim_groups

# old one
# 
def get_grouped_params_using_dimension_from_PD_0(
    model,
    weight_decay=0.0 # no decay as default
):
    assert isinstance(model, (ProteinDesigner_Base)), \
    "Only ProteinDesigner_Base is allowed for this fun."
    # start with all of the candidate parameters
    # this one may not work once the model is called
    # param_dict = {pn: p for pn, p in model.named_parameters()}
    # ++
#     # para in the top level
#     param_dict_top  = {pn: p for pn, p in model.named_parameters()}
    
#     # para in UNet level
#     param_dict_unet = {pn: p for pn, p in model.diffuser_core.unets[0].named_parameters()}
    
#     # merge the two
    
#     param_dict = param_dict_unet | param_dict_top
    
    param_dict  = {pn: p for pn, p in model.named_parameters()}
    
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    
    
    # create optim groups. 
    # Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, 
    # all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    # 
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    Print (f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    Print (f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    Print (f"In tot: {num_nodecay_params+num_decay_params}")

    return optim_groups
# 
# ==============================================================
# 
def configure_Adam_AdamW_optimizers(
    optim_groups,
    # opt parameters
    Adam_Or_AdamW,
    learning_rate,
    beta1,
    beta2,
    eps,
    **kwargs
):
    if Adam_Or_AdamW == 1:
        # Adam opt
        optimizer = Adam(
            optim_groups,
            lr = learning_rate,
            eps = eps,
            betas = (beta1, beta2),
            **kwargs
        )
    elif Adam_Or_AdamW == 2:
        # AdamW opt
        optimizer = AdamW(
            optim_groups,
            lr = learning_rate,
            betas = (beta1, beta2),
            **kwargs
        )

    return optimizer

# 
# ==============================================================
# 
# learning rate decay scheduler (cosine with warmup)
def get_lr_linear_cosine_contant(
    it,
    warmup_iters,
    lr_decay_iters,
    min_lr,
    learning_rate,
):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
        

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# on training: calculate loss for Protein Designer

@torch.no_grad()
def evaluate_loop_for_loss_of_PD(
    # 1. on model
    model,
    # 2. on data
    train_loader,
    max_batch_to_cover,
    # 3. 
    device
):
    '''
    1. calculate loss on test_loader
    '''
    
    assert isinstance(model, (ProteinDesigner_Base)), \
    "Working assuming it is a ProteinDesigner_Base object."
    
    assert max_batch_to_cover<= len(train_loader), \
    f"Ask for {max_batch_to_cover} batchs, but not enough batchs in the dataloder {len(train_loader)}"
    
    # # 1. put into eval() mode
    # model.eval()
    # model.diffuser_core.unets[0].eval()
    model.turn_on_eval_mode()
    
    # 2. calc loss for multi-batch
    # this loss is only for comparison
    loss_list = []
    for idx, item in enumerate(train_loader):
        if idx<max_batch_to_cover:
            
            X_train_batch = item[0].to(device) # (b, vib_mode, seq_len)
            Y_train_batch = item[1].to(device) # (b, seq_len)
            
            this_loss = model(
                Y_train_batch, # use seq_objs channel
                # 1. text condition via text_embed 
                text_con_input = X_train_batch, # to be mapped into text_con_embeds
                # 2. img condition 
                cond_images = X_train_batch,
            )
            
            loss_list.append(this_loss.item())
        else:
            break
    loss_mean = sum(loss_list)/len(loss_list)
    
    # # 3. put model back to training
    # model.train()
    # model.diffuser_core.unets[0].train()
    model.turn_on_train_mode()
        
    return loss_mean

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# on training: calculate loss for Protein Predictor

@torch.no_grad()
def evaluate_loop_for_loss_of_PP(
    # 1. on model
    model,
    # 2. on data
    train_loader,
    max_batch_to_cover,
    # 3. 
    device
):
    '''
    1. calculate loss on test_loader
    '''
    
    assert isinstance(model, (ProteinPredictor_Base)), \
    "Working assuming it is a ProteinPredictor_Base object."
    
    assert max_batch_to_cover<= len(train_loader), \
    f"Ask for {max_batch_to_cover} batchs, but not enough batchs in the dataloder {len(train_loader)}"
    
    # # 1. put into eval() mode
    # model.eval()
    # model.diffuser_core.unets[0].eval()
    model.turn_on_eval_mode()
    
    # 2. calc loss for multi-batch
    # this loss is only for comparison
    loss_list = []
    for idx, item in enumerate(train_loader):
        if idx<max_batch_to_cover:
            
            X_train_batch = item[0].to(device) # (b, vib_mode, seq_len)
            Y_train_batch = item[1].to(device) # (b, seq_len)
            
            this_loss = model(
                X_train_batch, # (b, vib_mode, seq_len) # use seq_objs channel
                # 1. text condition via text_embed 
                text_con_input = Y_train_batch, # to be mapped into text_con_embeds
                # 2. img condition 
                cond_images = Y_train_batch,
                #  
                unet_number = model.diffuser_core.only_train_unet_number
            )
            
            loss_list.append(this_loss.item())
        else:
            break
    loss_mean = sum(loss_list)/len(loss_list)
    
    # # 3. put model back to training
    # model.train()
    # model.diffuser_core.unets[0].train()
    model.turn_on_train_mode()
        
    return loss_mean

# 
# ==============================================================
# take dataloader as input
# assume both conditions and GT are available in the loader
# folding tool: OF
@torch.no_grad()
def sampling_loop_for_PP_on_DataLoder(
    # 1. on model
    model,
    # 2. on data,
    train_loader,
    max_batch_to_cover=2, # <= dataloader len
    max_seq_per_batch_to_fold=1,    # <= batch size
    noramfac_for_modes=None, # from PP_DataKeys
    # 3. on sampleing and folding
    cond_scales=[7.5], # between 5 and 10
    # 4. others
    device = None,
    CKeys = None,
    epoch = 'Test', # str(111),
    GAS = 'Test', # str(11),
    
    # 5. report contorlers
    sample_dir=None,
    sample_prefix='Sample', # "Trai" or "Test" # starter of all saved files
    # 5.1 on fig of toks 
    If_Plot_nms_vecs = True,
    IF_showfig = 1, # plot or save
    # 5.2 on basic Prediction package: Seq, PR, GTs
    IF_save_prediction_pack = 1,
    # # 5.3 on folding prediction via folding tools
    # IF_fold_seq = 1, # will produce PBD file
    # IF_show_foldding = 1, 
    # IF_DSSP = 1, # check SecStr
    
):
    assert isinstance(model, (ProteinPredictor_Base)), \
    "Working assuming it is a ProteinPredictor_Base object."
    
    
    # 1. put into eval() mode
    # model.eval()
    # model.diffuser_core.unets[0].eval()
    model.turn_on_eval_mode()
    n_mode = len(noramfac_for_modes)
    
    df_csv_filename = sample_dir + sample_prefix +\
    "_"+ str(GAS) + "_name_list.csv"
    # write the headline
    top_line = f"name_prefix,"
    for i_mode in range(n_mode):
        top_line += ("mode_"+str(i_mode+7)+"_rela_L2_err,")
    top_line += "multi_mode_rela_L2_err\n"
    
    write_one_line_to_file(
        this_line=top_line,
        file_name=df_csv_filename,
        mode = 'w',
    )
    
    # 2. 
    for i_batch, item in enumerate(train_loader):

        # only pick some mini-batchs
        if i_batch < max_batch_to_cover: 

            n_seq_in_this_batch = item[0].shape[0]
            # only pick some from one mini-batchs
            n_seq_pick = min(
                max_seq_per_batch_to_fold,
                n_seq_in_this_batch,
            )
            # 
            # # for Protein Designer
            # X_train_batch = item[0][:n_seq_pick].to(device) # (b, vib_mode, seq_len)
            # Y_train_batch = item[1][:n_seq_pick].to(device) # (b, seq_len)
            # 
            # for Protein Predictor
            X_train_batch = item[0][:n_seq_pick].to(device) # (b, vib_mode, seq_len)
            Y_train_batch = item[1][:n_seq_pick].to(device) # (b, seq_len)
            # b can be >= 1

            # GT for seq in toks
            # ..................................................
            # GT = Y_train_batch # .cpu().detach() # (b, seq_len=full)
            GT = X_train_batch # (b, vib_mode, seq_len)
            # translate into a list of toks
            
            # get len info as shape mask from seq toks (b, seq_len)
            mask_from_Y = model.read_mask_from_seq_toks_using_pLM(
                Y_train_batch
            ) 
            # ++
            if Local_Debug_Level == 1:
                print (f"GT: {type(GT)}")
                print (f"mask_from_X: {type(mask_from_X)}")
                # 
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # need to get the part of GT that is useful for error calc
            GT_nms_vecs_list = get_nms_vec_as_arr_list_from_batch_using_mask(
                mask_from_Y,  # (b, seq_len)
                GT,           # (b, vib_mode, seq_len)
                NormFac_list=noramfac_for_modes, # (vib_mode, )
            )
            # To store the Input Conditioning, (b, seq_len)
            # prep Y_train_batch into idx/toks
            Cond_AA_idx_arr_list, Cond_AA_string_list = model.map_batch_idx_into_AA_idx_and_toks_string_lists_w_pLM(
                Y_train_batch, # (b, seq_len)
                mask_from_Y,   # (b, seq_len)
            )
            # ++
            if Local_Debug_Level == 1:
                print (f"Cond_AA_idx_arr: {Cond_AA_idx_arr_list[0]}")
                print (f"Cond_AA_str: {Cond_AA_string_list[0]}")
            
            
            Cond_idx_arr_list, Cond_toks_string_list = model.map_batch_idx_into_AA_idx_and_toks_string_lists_w_pLM(
                Y_train_batch,
                mask_from_Y,
            )
            
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#             GT_toks_list, GT_seqs_list = get_toks_list_from_Y_batch(
#                 GT,
#                 mask_from_Y
#             )
            
#             # Reconstructed based on GT: measure recovability of pLM
#             # GT toks -> pLM logits -> 
#             # ..................................................
#             GT_logits = model.map_seq_tok_to_seq_channel_w_pLM(
#                 GT,
#             ) # (b, seq_len) --> (batch, num_toks, seq_len)
#             GT_logits = rearrange(GT_logits, 'b c l -> b l c')
#             # (batch, seq_len, num_toks)
#             GT_recon_toks_list, GT_recon_seqs_list = \
#             model.decode_many_esm_logits_w_mask(
#                 GT_logits, # (b, seq_len, num_toks)
#                 mask_from_Y, # (b, seq_len)
#                 common_AA_only=True,
#             )
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            
            for i_cond_scal, this_cond_scal in enumerate(cond_scales):
                # call sampling fun
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                PR_nms_vecs_list = model.sample_to_NMS_list(
                    # added ones
                    mask_from_Y=mask_from_Y, # (b, seq_len)
                    NormFac_list=noramfac_for_modes, # (n_mode)
                    # for self.sample
                    # ===============================
                    text_con_input=Y_train_batch, # (b, seq_len)
                    cond_images=Y_train_batch,    # (b, seq_len)
                    # 
                    cond_scale=this_cond_scal,
                )
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # PR_toks_list, PR_seqs_list, _ = model.sample_to_pLM_idx_seq(
                #     # 
                #     common_AA_only=True, # False,
                #     mask_from_Y=mask_from_Y, 
                #     # if none, will use mask from X, cond_img then text
                #     # 
                #     text_con_input = X_train_batch,
                #     cond_images = X_train_batch,
                #     # 
                #     cond_scale = this_cond_scal,
                # )
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                
                # compare with GT for reconverability
                for i_case in range(len(GT_nms_vecs_list)):
                    
                    this_head = f"{sample_prefix}_B_{i_batch}_C_{i_case}_CS_{str (this_cond_scal)}_{i_cond_scal}_at_{epoch}_{GAS}"
                    
                    Print (f"\nOn Case {this_head} ")
                    
                    outname_prefix = sample_dir + this_head
                    # +
                    this_csv_w_line = outname_prefix+','
    
                    
                    # 0. cheap report
                    # ........................................
                    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    # 0.1 ana the differences
                    dict_err_on_nms_vecs = compare_two_nms_vecs_arr(
                        PR_nms_vecs_list[i_case], # (n_mode, seq_len)
                        GT_nms_vecs_list[i_case], # (n_mode, seq_len)
                    )
                    # 0.2 print
                    block_to_print = f"""\
On Batch {i_batch}, Case {i_case}, Cond_scal: {this_cond_scal }
"""
                    Print (block_to_print)
                    for this_key in dict_err_on_nms_vecs.keys():
                        this_csv_w_line += str(dict_err_on_nms_vecs[this_key])+','
                        Print (f"{this_key}: {dict_err_on_nms_vecs[this_key]}")
                    # 
                    this_csv_w_line = this_csv_w_line[:-1]
                    # 
                    write_one_line_to_file(
                        this_line=this_csv_w_line+'\n',
                        file_name=df_csv_filename,
                        mode = 'a',
                    )
                    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                     # 0.1 ana seq difference
#                     hit_ratio_PR_GT = compare_two_seq_strings(
#                         PR_seqs_list[i_case],
#                         GT_seqs_list[i_case],
#                     )
#                     hit_ratio_GT_recon_GT = compare_two_seq_strings(
#                         GT_recon_seqs_list[i_case],
#                         GT_seqs_list[i_case],
#                     )
#                     # 0.2 print
#                     block_to_print = f"""\
# On Batch {i_batch}, Case {i_case}, Cond_scal: {this_cond_scal }
# Seq pair: PR, GT, GT_recon_pLM
# {PR_seqs_list[i_case]}
# {GT_seqs_list[i_case]}
# {GT_recon_seqs_list[i_case]}
# recovery ratio: PR-GT for model, GT_recon-GT for pLM
# {hit_ratio_PR_GT}, {hit_ratio_GT_recon_GT}
#                     """
#                     Print (block_to_print)
                    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    
                    
                    # 1. medium report: plot
                    # ........................................
                    if If_Plot_nms_vecs:
                        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                        Print (f"Plot the modes...")
                        
                        fig=plt.figure()
                        for i_mode in range(n_mode):
                            # PR
                            plt.plot (
                                PR_nms_vecs_list[i_case][i_mode],
                                '^-',
                                label= f'Predicted mode {i_mode}',
                            )
                            # GT
                            plt.plot (
                                GT_nms_vecs_list[i_case][i_mode],
                                '--',
                                label= f'GT mode {i_mode}',
                            )
                        
                        plt.legend()
                        plt.xlabel(f"AA #")
                        plt.ylabel(f"Vibrational Disp. amp")
                        outname = outname_prefix+"_VibDisAmp.jpg"
                        if IF_showfig==1:
                            plt.show()
                        else:
                            plt.savefig(outname, dpi=200)
                        plt.close ()
                        outname = None
                        
                        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        
#                         Print (f"Plot sequence idx...\n")
                        
#                         fig=plt.figure()
#                         plt.plot (
#                             PR_toks_list[i_case].cpu().detach().numpy(),
#                             '^-',
#                             label= f'Predicted',
#                         )
#                         plt.plot (
#                             GT_toks_list[i_case].cpu().detach().numpy(),
#                             label= f'GT'
#                         )
#                         plt.plot (
#                             GT_recon_toks_list[i_case].cpu().detach().numpy(),
#                             '--',
#                             label= f'GT-pLM'
#                         )
#                         plt.legend()
#                         plt.xlabel(f"AA #")
#                         plt.ylabel(f"idx in ESM")
#                         outname = outname_prefix+"_comp.jpg"
#                         if IF_showfig==1:
#                             plt.show()
#                         else:
#                             plt.savefig(outname, dpi=200)
#                         plt.close ()
#                         outname = None
                        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                        
                    # 2. save basics: input X, GT Y and PR Y
                    # ........................................
                    if IF_save_prediction_pack == 1:
                        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                        # 2.1 on the input X: idx_arr
                        outname = outname_prefix+"_XCond_Seq_idx.txt"
                        # Cond_AA_idx_arr_list
                        np.savetxt(
                            outname,
                            Cond_idx_arr_list[i_case],
                        )
                        # ++
                        if Local_Debug_Level==1:
                            test_Cond_idx_arr = np.loadtxt(outname)
                            print (f"Read back Cond_idx_arr: \n{test_Cond_idx_arr}")
                        # 
                        outname=None
                        # 
                        # 2.2 on the input X: Seq_string
                        outname = outname_prefix+"_XCond_Seq_string.fasta"
                        this_label_list = [
                            f"XCond"
                        ]
                        this_seq_list = [
                            Cond_toks_string_list[i_case]
                        ]
                        write_fasta_file(
                            this_seq_list=this_seq_list, 
                            this_head_list=this_label_list, 
                            this_file=outname
                        ) # can handle multiple seqs in one fasta file
                        outname = None
                        # ++
                        if Local_Debug_Level==1:
                            print (f"Cond_toks_string: \n{Cond_toks_string_list[i_case]}")
                        #
                        # 2.3 on the output: GT
                        outname = outname_prefix+"_GT.txt"
                        np.savetxt(
                            outname,
                            GT_nms_vecs_list[i_case]
                        )
                        outname = None
                        
                        # 2.4 on the output: PR
                        outname = outname_prefix+"_PR.txt"
                        np.savetxt(
                            outname,
                            PR_nms_vecs_list[i_case]
                        )
                        outname = None
                        
                        
                        
                        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                         # 2.1 on the input X, applying mask for Y
#                         outname = outname_prefix+"_XCond.txt"
#                         save_2d_tensor_as_np_arr_txt(
#                             X_train_batch[i_case], # .cpu().detach().numpy(),
#                             mask = mask_from_Y[i_case], # set to None to turn off
#                             outname = outname,
#                         )
#                         # ++
#                         if Local_Debug_Level==1:
#                             this_X_cond_arr = np.loadtxt(outname)
#                             print (f"X_cond_arr.shape: {this_X_cond_arr.shape}")
                            
#                         outname = None
#                         # 2.2 on GT Y: sequence
#                         outname = outname_prefix+"_GT.fasta"
#                         this_label_list = [
#                             f"GT, Reconvery ratio: GT_con_GT: {hit_ratio_GT_recon_GT}",
#                             f"GT_recon, Reconvery ratio: GT_con_GT: {hit_ratio_GT_recon_GT}"
#                         ]
#                         this_seq_list = [
#                             GT_seqs_list[i_case],
#                             GT_recon_seqs_list[i_case]
#                         ]
#                         write_fasta_file(
#                             this_seq_list=this_seq_list, 
#                             this_head_list=this_label_list, 
#                             this_file=outname
#                         ) # can handle multiple seqs in one fasta file
#                         outname = None
#                         # 2.3 on PR: sequence
#                         this_label = f"Recovery ratio: PR_GT: {hit_ratio_PR_GT}, GT_con_GT: {hit_ratio_GT_recon_GT}"
#                         outname = outname_prefix+"_PR.fasta"
#                         write_fasta_file(
#                             this_seq_list=[PR_seqs_list[i_case]], 
#                             this_head_list=[this_label], 
#                             this_file=outname
#                         )
#                         outname = None
                        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                        
                    # 3. expansive one: folding prediction and show
                    # skipped here
                
                    # 4. write key results to a d
                            
                        
    
    
    # 3. put model back to training
    # model.train()
    # model.diffuser_core.unets[0].train()
    model.turn_on_train_mode()
    
    
    return df_csv_filename


# 
# ==============================================================
# take list of conditions as input
# no GT is available
# for small number of input at this time, may be improved later
@torch.no_grad()
def sampling_loop_for_PP_on_ConditionList(
    # 1. on model
    model,
    # 2. on input conditioning
    raw_condition_list, # list of (AA, ) for PP # list of (mode=3, seq_len) np arr FOR PD
    AA_seq_max_len=None, # Used for PP
    noramfac_for_modes=None, # used for PP
    sample_batch_size=256,
    cond_scales=[7.5], # between 5 and 10
    # 3. on sampling controlers
    sample_dir=None,
    If_print_vecs=True,    # print the predicted NMS vecs
    If_Plot_vecs=True,     # generate plot of seq idx
    IF_showfig=1, # show the plot
    IF_save_prediction_pack=1, # save seq in fasta
    # IF_fold_seq = 1, # fold AA into PDB
    # IF_show_foldding=1, # visualization pd
    # IF_DSSP=1, # ana. SecStr
    # 4. others
    sample_prefix="DeNovo",
    epoch='Test',
    GAS='Test',
    device=None,
):
    # 0. check base line
    assert isinstance(model, (ProteinPredictor_Base)), \
    "Working assuming it is a ProteinDesigner_Base object."
    
    # 1. mode update
    model.turn_on_eval_mode()
    n_mode = model.seq_obj_channels # 3
    
    # 2.prepare input
    
    # put conditioning input into one batch
    # borrow from DataSetPack.py
    
    # 2.1 get Y data padded
    # translate AA list to a batch of idx, like what we have at dataloader
    
    # need to save 2 positions for <cls> and <eos>
    esm_batch_converter = model.pLM_alphabet.get_batch_converter(
        truncation_seq_length=AA_seq_max_len-2
    )
    model.pLM.eval() # disables dropout for deterministic results
    # prepare seqs for the "esm_batch_converter..."
    # ++ add dummy labels
    seqs_ext=[]
    # add a fake one to make sure the padding length
    dummy_seq = 'A'*(AA_seq_max_len-2)
    seqs_ext.append(
        (" ", dummy_seq)
    )
    
    for i in range(len(raw_condition_list)):
        seqs_ext.append(
            (" ", raw_condition_list[i])
        )
    # batch_labels, batch_strs, batch_tokens = esm_batch_converter(seqs_ext)
    _, y_strs, y_data = esm_batch_converter(seqs_ext)
    # y_strs_lens = (y_data != esm_alphabet.padding_idx).sum(1)
    # print(batch_tokens.shape)
    # 
    # ++ remove the dummy one
    y_data = y_data[1:]
    seqs_ext = seqs_ext[1:]
    print ("y_data.dim: ", y_data.dtype)
    
    # only for test, turn off later
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if Local_Debug_Level==1:
        # assume y_data is reversiable
        y_data_reversed = decode_many_ems_token_rec(y_data, model.pLM_alphabet)
        # check by pick some
        for iii in [0,1,]:
            print("Ori and REVERSED SEQ: ", iii)
            print(raw_condition_list[iii])
            print(y_data_reversed[iii])
        
    # 2.2 get Y normalized and formated
    # not used for AA
    
    # 2.3 process X_train as mini-batchs
    Y_train_batch_list = torch.split(y_data, sample_batch_size)
    n_batch = len(Y_train_batch_list)
    
    
    X_file_list = []
    PR_file_list = []
    pdb_file_list = []
    
    for i_cond_scal, this_cond_scal in enumerate(cond_scales):
        
        jj_case = -1 # to mark the seq
        for i_batch in range(n_batch):
            
            Y_train_batch = Y_train_batch_list[i_batch].to(device)
            # (b, seq)
            Print (f"this batch, Y.shape: {Y_train_batch.shape}")
            
            # 2.4 call sampling fun
            
            mask_from_AA = model.read_mask_from_seq_toks_using_pLM(
                Y_train_batch
            ) # (b, seq_len)
            
            PR_NMS_arr_list = model.sample_to_NMS_list(
                # mask
                mask_from_Y = mask_from_AA,
                NormFac_list = noramfac_for_modes,
                # 
                text_con_input = Y_train_batch,
                cond_images = Y_train_batch,
                # 
                cond_scale = this_cond_scal,
            ) # list (n_mode, seq_len)
            
            
            # 2.5 postprocess per sample in this batch
            for i_case in range(len(Y_train_batch)):
                jj_case += 1
                
                idx_sample = i_batch*sample_batch_size+i_case
                this_head = f"{sample_prefix}_{idx_sample}_CS_{str (this_cond_scal)}_{i_cond_scal}_at_{epoch}_{GAS}"
                
                outname_prefix = sample_dir + this_head
                
                # 0. cheap report
                # ........................................
                # 0.1 ana seq difference: skipped for no GT
                
                # 0.2 print: skipped
                block_to_print = f"""\
On Batch {i_batch}, Case {i_case}, Cond_scal: {this_cond_scal }
NMS vec: PR
{PR_NMS_arr_list[i_case].shape}
"""
                if If_print_vecs:
                    Print (block_to_print)
                
                # 1. medium report: plot
                # ........................................
                if If_Plot_vecs:

                    Print (f"Compare predicted NMS vecs...\n")

                    fig=plt.figure()
                    
                    for i_mode in range(n_mode):
                        plt.plot (
                            PR_NMS_arr_list[i_case][i_mode],
                            '^-',
                            label= f'Predicted mode {i_mode}',
                        )
                        
                    plt.legend()
                    plt.xlabel(f"AA #")
                    plt.ylabel(f"Vibrational Disp. amp")
                    outname = outname_prefix+"_VibDisAmp.jpg"
                    if IF_showfig==1:
                        plt.show()
                    else:
                        plt.savefig(outname, dpi=200)
                    plt.close ()
                    outname = None
                    
                # 2. save basics: input X, GT Y and PR Y
                # ........................................
                if IF_save_prediction_pack == 1:
                    # 2.1 on the input X, applying mask for Y
                    outname = outname_prefix+"_XCond_Seq.fasta"
                    this_label_list = [
                        f"XCond"
                    ]
                    this_seq_list = [
                        raw_condition_list[jj_case]
                    ]
                    write_fasta_file(
                        this_seq_list=this_seq_list, 
                        this_head_list=this_label_list, 
                        this_file=outname
                    ) # can handle multiple seqs in one fasta file
                    # 
                    X_file_list.append(outname)
                    Print(f"Input X:\n{raw_condition_list[jj_case]}\n")
                    # ++
                    if Local_Debug_Level==1:
                        print (f"this input seq: {raw_condition_list[jj_case]}")
                    # 
                    outname = None
                    
                    # 2.2 on GT Y: Normal mode vectors: skiiped

                    # 2.3 on PR: Normal mode vectors
                    outname = outname_prefix+"_PR.txt"
                    np.savetxt(
                        outname, 
                        PR_NMS_arr_list[i_case]
                    )
                    # +++
                    PR_file_list.append(outname)
                    outname = None
                    
                # 3. expansive one: folding prediction and show
                # ........................................
                # skipped herr
        
    
    # 3. mode update
    model.turn_on_train_mode()
    
    return X_file_list, PR_file_list

# 
# ==============================================================
# take list of conditions as input
# no GT is available
# for small number of input at this time, may be improved later
@torch.no_grad()
def sampling_loop_for_PD_on_ConditionList_w_OF(
    # 1. on model
    model,
    # 2. on input conditioning
    raw_condition_list, # list of (mode=3, seq_len) np arr
    noramfac_for_modes=None,
    sample_batch_size=256,
    cond_scales=[7.5], # between 5 and 10
    # 3. on sampling controlers
    sample_dir=None,
    If_Plot_seq_toks=True, # generate plot of seq idx
    IF_showfig=1, # show the plot
    IF_save_prediction_pack=1, # save seq in fasta
    IF_fold_seq = 1, # fold AA into PDB
    IF_show_foldding=1, # visualization pd
    IF_DSSP=1, # ana. SecStr
    # 4. others
    sample_prefix="DeNovo",
    epoch='Test',
    GAS='Test',
    device=None,
):
    # 0. check base line
    assert isinstance(model, (ProteinDesigner_Base)), \
    "Working assuming it is a ProteinDesigner_Base object."
    
    # 1. mode update
    model.turn_on_eval_mode()
    
    # 2.prepare input
    
    # put conditioning input into one batch
    # borrow from DataSetPack.py
    
    # 2.1 get X data padded
    text_len_max = model.text_max_len
    img_len_max = model.diffuser_core.image_sizes[0]
    len_in = min(text_len_max, img_len_max) # depend on problem statement, may change
    
    text_embed_input_dim = model.text_embed_input_dim
    cond_img_channels = model.diffuser_core.unets[0].cond_images_channels
    mode_in = min(text_embed_input_dim, cond_img_channels)
    
    n_in = len(raw_condition_list)
    
    X = np.zeros(
        (n_in, mode_in, len_in)
    )
    for i in range(n_in):
        for j in range(mode_in):
            X[i, j, :] = pad_a_np_arr_esm_for_NMS(
                raw_condition_list[i][j,:],
                0,
                len_in
            )
    Print(f"Pack inputs, X.shape: {X.shape}")
    
    # 2.2 get X normalized and formated
    for j in range(mode_in):
        X[:,j,:] = X[:,j,:]/noramfac_for_modes[j]
        
    X_train = torch.from_numpy(X).float() # (b, c, seq_len)
        
    # 2.3 process X_train as mini-batchs
    X_train_batch_list = torch.split(X_train, sample_batch_size)
    n_batch = len(X_train_batch_list)
    
    X_file_list = []
    pdb_file_list = []
    fasta_file_list = []
    
    for i_cond_scal, this_cond_scal in enumerate(cond_scales):
        
        for i_batch in range(n_batch):
            X_train_batch = X_train_batch_list[i_batch].to(device)
            # (b, c, seq)
            Print (f"this batch, X.shape: {X_train_batch.shape}")
            
            # 2.4 call sampling fun
            # will get shape-mask from X_train_batch
            PR_toks_list, PR_seqs_list, result_mask = \
            model.sample_to_pLM_idx_seq(
                # 
                common_AA_only=True, # False,
                mask_from_Y=None, # mask_from_Y, 
                # if none, will use mask from X, cond_img then text
                # 
                text_con_input = X_train_batch,
                cond_images = X_train_batch,
                # 
                cond_scale = this_cond_scal,
            )
            
            # 2.5 postprocess per sample in this batch
            for i_case in range(len(X_train_batch)):
                idx_sample = i_batch*sample_batch_size+i_case
                this_head = f"{sample_prefix}_{idx_sample}_CS_{str (this_cond_scal)}_{i_cond_scal}_at_{epoch}_{GAS}"
                
                outname_prefix = sample_dir + this_head
                
                # 0. cheap report
                # ........................................
                # 0.1 ana seq difference: skipped for no GT
                
                # 0.2 print
                block_to_print = f"""\
On Batch {i_batch}, Case {i_case}, Cond_scal: {this_cond_scal }
Seq pair: PR
{PR_seqs_list[i_case]}
"""
                Print (block_to_print)
                
                # 1. medium report: plot
                # ........................................
                if If_Plot_seq_toks:

                    Print (f"Compare sequence idx...\n")

                    fig=plt.figure()
                    plt.plot (
                        PR_toks_list[i_case].cpu().detach().numpy(),
                        '^-',
                        label= f'Predicted',
                    )
                    plt.legend()
                    plt.xlabel(f"AA #")
                    plt.ylabel(f"idx in ESM")
                    outname = outname_prefix+"_comp.jpg"
                    if IF_showfig==1:
                        plt.show()
                    else:
                        plt.savefig(outname, dpi=200)
                    plt.close ()
                    outname = None
                    
                # 2. save basics: input X, GT Y and PR Y
                # ........................................
                if IF_save_prediction_pack == 1:
                    # 2.1 on the input X, applying mask for Y
                    outname = outname_prefix+"_XCond.txt"
                    save_2d_tensor_as_np_arr_txt(
                        X_train_batch[i_case], # .cpu().detach().numpy(),
                        mask = result_mask[i_case], # mask_from_Y[i_case], 
                        # set to None to turn off
                        outname = outname,
                    )
                    X_file_list.append(outname)
                    # ++
                    if Local_Debug_Level==1:
                        this_X_cond_arr = np.loadtxt(outname)
                        print (f"X_cond_arr.shape: {this_X_cond_arr.shape}")

                    outname = None
                    
                    # 2.2 on GT Y: sequence: skiiped

                    # 2.3 on PR: sequence
                    this_label = f"{sample_prefix}"
                    outname = outname_prefix+"_PR.fasta"
                    write_fasta_file(
                        this_seq_list=[PR_seqs_list[i_case]], 
                        this_head_list=[this_label], 
                        this_file=outname
                    )
                    # +++
                    fasta_file_list.append(outname)
                    outname = None
                    
                # 3. expansive one: folding prediction and show
                # ........................................
                if IF_fold_seq == 1:

                    Print (f"Folding the predicted AA...")

                    outname = outname_prefix+"_PR.pdb"

                    # 3.1 fold AA using Omegaofld
                    # ........................................
                    PR_PDB_temp, PR_AA_temp = \
                    fold_one_AA_to_SS_using_omegafold(
                        sequence=PR_seqs_list[i_case],
                        num_cycle=16,
                        device=device,
                        # ++++++++++++++
                        prefix="Temp", # None,
                        AA_file_path=sample_dir, # "./",  # None,  
                        PDB_file_path=sample_dir, # "./", # output file path
                        head_note="Temp_", # None,
                    ) 
                    # THe pdb file name is decided by the AA head_note
                    # Not easy to put path there.
                    # so, here we copy it.
                    shutil.copy(PR_PDB_temp,outname) # (sour, dest)
                    # don't need fasta. But need to clean it up
                    # clean the slade to avoid mistakenly using the previous fasta file
                    os.remove (PR_PDB_temp) # 
                    os.remove (PR_AA_temp)
                    # +++
                    pdb_file_list.append(outname)
                    
                    # 3.2 visualize the chain for pLDDT
                    # ........................................
                    if IF_show_foldding==1:

                        Print (f"Show the folded structure...")
                        Print (f"Only use this in a non-silent running...")

                        view=show_pdb(
                            pdb_file=outname, 
                            # flag=flag,
                            show_sidechains=False, #choose from {type:"boolean"} show_sidechains, 
                            show_mainchains=False, #choose from {type:"boolean"} show_mainchains, 
                            color= "lDDT", # choose from ["chain", "lDDT", "rainbow"]color
                        )
                        view.show()
                    
                    # 3.3 post analysis
                    # ........................................
                    if IF_DSSP == 1:

                        Print (f"Ana. secondary structure...")
                        # apply dssp
                        PR_DSSP_Q8, PR_DSSP_Q3, PR_AA_from_DSSP = \
                        get_DSSP_result(
                            outname
                        )
                        # check if any AA is missing
                        if len(PR_AA_from_DSSP)!=len(PR_seqs_list[i_case]):
                            Print (f"Missing AA during DSSP. Use caution.")
                        # print result
                        block_to_print = f"""\
AA:           {PR_seqs_list[i_case]}
AA from DSSP: {PR_AA_from_DSSP}
Q8:           {PR_DSSP_Q8}
Q3:           {PR_DSSP_Q3}
"""
                        Print (block_to_print)

                        # save to file
                        outname_DSSP = outname_prefix+"_DSSP.json"
                        write_DSSP_result_to_json(
                            # content
                            PR_DSSP_Q8,
                            PR_DSSP_Q3,
                            PR_AA_from_DSSP,
                            # file
                            outname_DSSP,
                        )

                    # 3.4 clean up those depends on PDB
                    # ........................................
                    outname = None
                    outname_DSSP = None
        
        
    
    # 3. mode update
    model.turn_on_train_mode()
    
    return X_file_list, pdb_file_list, fasta_file_list
    
# 
# ==============================================================
# take dataloader as input
# assume both conditions and GT are available in the loader
# folding tool: OF
@torch.no_grad()
def sampling_loop_for_PD_on_DataLoder_w_OF(
    # 1. on model
    model,
    # 2. on data,
    train_loader,
    max_batch_to_cover=2, # <= dataloader len
    max_seq_per_batch_to_fold=1,    # <= batch size
    # 3. on sampleing and folding
    cond_scales=[7.5], # between 5 and 10
    # 4. others
    device = None,
    CKeys = None,
    epoch = 'Test', # str(111),
    GAS = 'Test', # str(11),
    
    # 5. report contorlers
    sample_dir=None,
    sample_prefix='Sample', # "Trai" or "Test" # starter of all saved files
    # 5.1 on fig of toks 
    If_Plot_seq_toks = True,
    IF_showfig = 1, # plot or save
    # 5.2 on basic Prediction package: Seq, PR, GTs
    IF_save_prediction_pack = 1,
    # 5.3 on folding prediction via folding tools
    IF_fold_seq = 1, # will produce PBD file
    IF_show_foldding = 1, 
    IF_DSSP = 1, # check SecStr
    
):
    assert isinstance(model, (ProteinDesigner_Base)), \
    "Working assuming it is a ProteinDesigner_Base object."
    
    
    # 1. put into eval() mode
    # model.eval()
    # model.diffuser_core.unets[0].eval()
    model.turn_on_eval_mode()
    
    # 2. 
    for i_batch, item in enumerate(train_loader):

        # only pick some mini-batchs
        if i_batch < max_batch_to_cover: 

            n_seq_in_this_batch = item[0].shape[0]
            # only pick some from one mini-batchs
            n_seq_pick = min(
                max_seq_per_batch_to_fold,
                n_seq_in_this_batch,
            )
            # 
            X_train_batch = item[0][:n_seq_pick].to(device) # (b, vib_mode, seq_len)
            Y_train_batch = item[1][:n_seq_pick].to(device) # (b, seq_len)
            # b can be >= 1

            # GT for seq in toks
            # ..................................................
            GT = Y_train_batch # .cpu().detach() # (b, seq_len=full)
            # translate into a list of toks
            
            # get len info as shape mask
            mask_from_Y = model.read_mask_from_seq_toks_using_pLM(
                Y_train_batch
            )
            # ++
            if Local_Debug_Level == 1:
                print (f"GT: {type(GT)}")
                print (f"mask_from_Y: {type(mask_from_Y)}")
            GT_toks_list, GT_seqs_list = get_toks_list_from_Y_batch(
                GT,
                mask_from_Y
            )
            
            # Reconstructed based on GT: measure recovability of pLM
            # GT toks -> pLM logits -> 
            # ..................................................
            GT_logits = model.map_seq_tok_to_seq_channel_w_pLM(
                GT,
            ) # (b, seq_len) --> (batch, num_toks, seq_len)
            GT_logits = rearrange(GT_logits, 'b c l -> b l c')
            # (batch, seq_len, num_toks)
            GT_recon_toks_list, GT_recon_seqs_list = \
            model.decode_many_esm_logits_w_mask(
                GT_logits, # (b, seq_len, num_toks)
                mask_from_Y, # (b, seq_len)
                common_AA_only=True,
            )
            
            
            for i_cond_scal, this_cond_scal in enumerate(cond_scales):
                # call sampling fun
                PR_toks_list, PR_seqs_list, _ = model.sample_to_pLM_idx_seq(
                    # 
                    common_AA_only=True, # False,
                    mask_from_Y=mask_from_Y, 
                    # if none, will use mask from X, cond_img then text
                    # 
                    text_con_input = X_train_batch,
                    cond_images = X_train_batch,
                    # 
                    cond_scale = this_cond_scal,
                )
                
                # compare with GT for reconverability
                for i_case in range(len(GT_toks_list)):
                    
                    this_head = f"{sample_prefix}_B_{i_batch}_C_{i_case}_CS_{str (this_cond_scal)}_{i_cond_scal}_at_{epoch}_{GAS}"
                    
                    Print (f"\nOn Case {this_head} ")
                    
                    outname_prefix = sample_dir + this_head
                    
                    # 0. cheap report
                    # ........................................
                    # 0.1 ana seq difference
                    hit_ratio_PR_GT = compare_two_seq_strings(
                        PR_seqs_list[i_case],
                        GT_seqs_list[i_case],
                    )
                    hit_ratio_GT_recon_GT = compare_two_seq_strings(
                        GT_recon_seqs_list[i_case],
                        GT_seqs_list[i_case],
                    )
                    # 0.2 print
                    block_to_print = f"""\
On Batch {i_batch}, Case {i_case}, Cond_scal: {this_cond_scal }
Seq pair: PR, GT, GT_recon_pLM
{PR_seqs_list[i_case]}
{GT_seqs_list[i_case]}
{GT_recon_seqs_list[i_case]}
recovery ratio: PR-GT for model, GT_recon-GT for pLM
{hit_ratio_PR_GT}, {hit_ratio_GT_recon_GT}
                    """
                    Print (block_to_print)
                    
                    # Print (f"On Batch {i_batch}, Case {i_case}, Cond_scal: {this_cond_scal },")
                    # Print (f"Seq pair: PR, GT, GT_recon_pLM")
                    # Print (f"{PR_seqs_list[i_case]}")
                    # Print (f"{GT_seqs_list[i_case]}")
                    # Print (f"{GT_recon_seqs_list[i_case]}")
                    # Print (f"recovery ratio: PR-GT for model, GT_recon-GT for pLM")
                    # Print (f"{hit_ratio_PR_GT}, {hit_ratio_GT_recon_GT}")
                    
                    
                    # 1. medium report: plot
                    # ........................................
                    if If_Plot_seq_toks:
                        
                        Print (f"Plot sequence idx...\n")
                        
                        fig=plt.figure()
                        plt.plot (
                            PR_toks_list[i_case].cpu().detach().numpy(),
                            '^-',
                            label= f'Predicted',
                        )
                        plt.plot (
                            GT_toks_list[i_case].cpu().detach().numpy(),
                            label= f'GT'
                        )
                        plt.plot (
                            GT_recon_toks_list[i_case].cpu().detach().numpy(),
                            '--',
                            label= f'GT-pLM'
                        )
                        plt.legend()
                        plt.xlabel(f"AA #")
                        plt.ylabel(f"idx in ESM")
                        outname = outname_prefix+"_comp.jpg"
                        if IF_showfig==1:
                            plt.show()
                        else:
                            plt.savefig(outname, dpi=200)
                        plt.close ()
                        outname = None
                        
                    # 2. save basics: input X, GT Y and PR Y
                    # ........................................
                    if IF_save_prediction_pack == 1:
                        # 2.1 on the input X, applying mask for Y
                        outname = outname_prefix+"_XCond.txt"
                        save_2d_tensor_as_np_arr_txt(
                            X_train_batch[i_case], # .cpu().detach().numpy(),
                            mask = mask_from_Y[i_case], # set to None to turn off
                            outname = outname,
                        )
                        # ++
                        if Local_Debug_Level==1:
                            this_X_cond_arr = np.loadtxt(outname)
                            print (f"X_cond_arr.shape: {this_X_cond_arr.shape}")
                            
                        outname = None
                        # 2.2 on GT Y: sequence
                        outname = outname_prefix+"_GT.fasta"
                        this_label_list = [
                            f"GT, Reconvery ratio: GT_con_GT: {hit_ratio_GT_recon_GT}",
                            f"GT_recon, Reconvery ratio: GT_con_GT: {hit_ratio_GT_recon_GT}"
                        ]
                        this_seq_list = [
                            GT_seqs_list[i_case],
                            GT_recon_seqs_list[i_case]
                        ]
                        write_fasta_file(
                            this_seq_list=this_seq_list, 
                            this_head_list=this_label_list, 
                            this_file=outname
                        ) # can handle multiple seqs in one fasta file
                        outname = None
                        # 2.3 on PR: sequence
                        this_label = f"Recovery ratio: PR_GT: {hit_ratio_PR_GT}, GT_con_GT: {hit_ratio_GT_recon_GT}"
                        outname = outname_prefix+"_PR.fasta"
                        write_fasta_file(
                            this_seq_list=[PR_seqs_list[i_case]], 
                            this_head_list=[this_label], 
                            this_file=outname
                        )
                        outname = None
                        
                    # 3. expansive one: folding prediction and show
                    # ........................................
                    if IF_fold_seq == 1:
                        
                        Print (f"Folding the predicted AA...")
                        
                        outname = outname_prefix+"_PR.pdb"
                        
                        # 3.1 fold AA using Omegaofld
                        # ........................................
                        PR_PDB_temp, PR_AA_temp = \
                        fold_one_AA_to_SS_using_omegafold(
                            sequence=PR_seqs_list[i_case],
                            num_cycle=16,
                            device=device,
                            # ++++++++++++++
                            prefix="Temp", # None,
                            AA_file_path=sample_dir, # "./",  # None,  
                            PDB_file_path=sample_dir, # "./", # output file path
                            head_note="Temp_", # None,
                        ) 
                        # THe pdb file name is decided by the AA head_note
                        # Not easy to put path there.
                        # so, here we copy it.
                        shutil.copy(PR_PDB_temp,outname) # (sour, dest)
                        # don't need fasta. But need to clean it up
                        # clean the slade to avoid mistakenly using the previous fasta file
                        os.remove (PR_PDB_temp) # 
                        os.remove (PR_AA_temp)
                        
                        # 3.2 visualize the chain for pLDDT
                        # ........................................
                        if IF_show_foldding==1:
                            
                            Print (f"Show the folded structure...")
                            Print (f"Only use this in a non-silent running...")
                            
                            view=show_pdb(
                                pdb_file=outname, 
                                # flag=flag,
                                show_sidechains=False, #choose from {type:"boolean"} show_sidechains, 
                                show_mainchains=False, #choose from {type:"boolean"} show_mainchains, 
                                color= "lDDT", # choose from ["chain", "lDDT", "rainbow"]color
                            )
                            view.show()
                            
                        # 3.3 post analysis
                        # ........................................
                        if IF_DSSP == 1:
                            
                            Print (f"Ana. secondary structure...")
                            # apply dssp
                            PR_DSSP_Q8, PR_DSSP_Q3, PR_AA_from_DSSP = \
                            get_DSSP_result(
                                outname
                            )
                            # check if any AA is missing
                            if len(PR_AA_from_DSSP)!=len(PR_seqs_list[i_case]):
                                Print (f"Missing AA during DSSP. Use caution.")
                            # print result
                            block_to_print = f"""\
AA:           {PR_seqs_list[i_case]}
AA from DSSP: {PR_AA_from_DSSP}
Q8:           {PR_DSSP_Q8}
Q3:           {PR_DSSP_Q3}
"""
                            Print (block_to_print)
                            
                            # save to file
                            outname_DSSP = outname_prefix+"_DSSP.json"
                            write_DSSP_result_to_json(
                                # content
                                PR_DSSP_Q8,
                                PR_DSSP_Q3,
                                PR_AA_from_DSSP,
                                # file
                                outname_DSSP,
                            )
                            
                        # 3.4 clean up those depends on PDB
                        # ........................................
                        outname = None
                        outname_DSSP = None

                            
                        
                        
                
                
                
            
    
    
    
    
    # 3. put model back to training
    # model.train()
    # model.diffuser_core.unets[0].train()
    model.turn_on_train_mode()
    
    
    return

# on save the last few files

def keep_only_the_latest_N_files(
    folder_path,
    file_name_starter,
    num_to_keep
):
    """
    Keeps the latest N files in a folder and deletes the rest.
    """

    files = []
    # save_steps = []
    len_starter = len(file_name_starter)
    for entry in os.scandir(folder_path):
        if entry.is_file():
            if entry.name[:len_starter]==file_name_starter:
                
                files.append((
                    entry.path, 
                    entry.stat().st_mtime,
                    int(entry.name.split('_')[2])
                ))
    
    # # 1. if use modification time
    # # Sort files by modification time (newest first)
    # files.sort(key=lambda x: x[1], reverse=True)
    # 2. if use saved steps
    files.sort(key=lambda x: x[2], reverse=True)

    # Delete older files
    if len(files)>num_to_keep:
        for file_path, _, _ in files[num_to_keep:]:
            os.remove(file_path)

# 
# ==============================================================
# training loop: Protein Designer
# 
def train_loop_for_PD_on_DataLoder(
    # 0. basic
    train_loader_noshuffle,
    test_loader,
    wk_ProteinDesigner, # model
    optimizer,
    TrainKeys,
    CKeys,
    device,
    # 1. pick-up from previous
    finished_steps = None, # finished_steps_0 # end of the previous training point
    completed_updating_steps = None, # completed_updating_steps_0 # GAS finished previously
    best_val_loss = None, # best_val_loss_0
    GAS_at_best_val_loss = None, # GAS_at_best_val_loss_0
    # 2. freq to check and save
    gradient_accumulation_steps = None, # TrainKeys['gradient_accumulation_steps']
    last_num_ckpt_to_keep = 2,
    # 2.1 on eval_loss
    eval_during_trai_1_max_batch_to_cover = 5,
    # 2.2 on sampling on test dataloader
    sampling_max_batch_to_cover = 2,
    sampling_max_seq_per_batch_to_fold = 2,
    sampling_cond_scales = [7.5],
    sampling_If_Plot_seq_toks = True,
    sampling_IF_save_prediction_pack = 1,
    sampling_IF_fold_seq = 1,
    sampling_IF_DSSP = 1,
    sampling_sample_prefix = 'Trai_Samp',
    # 2.3 on denovo sampling
    denovo_raw_condition_list = None, # raw_condition_list
    noramfac_for_modes = None,
    denovo_cond_scales = [7.5], 
    denovo_sample_batch_size = 256,
    denovo_If_Plot_seq_toks = True,
    denovo_IF_save_prediction_pack = 1,
    denovo_IF_fold_seq = 1,
    denovo_IF_DSSP = 1,
    denovo_sample_prefix = 'Trai_Denovo',
    
):
    
    Print(f"Previously, ")
    Print(f" finished_steps_0: {finished_steps}")
    Print(f" completed_updating_steps_0: {completed_updating_steps}")
    Print(f" best_val_loss_0: {best_val_loss}")
    
    
    # prepare
    target_epochs = TrainKeys['num_train_epochs']
    
    wk_ProteinDesigner.train()
    # to make sure: direct talk to unets
    wk_ProteinDesigner.diffuser_core.unets[0].train()
    # clean
    optimizer.zero_grad(set_to_none=True)
    
    this_step = 0 # real mini-batch counter
    for epoch in range(target_epochs):
        for item in train_loader_noshuffle:
            this_step += 1 # start from 1, not 0
            if this_step > finished_steps: # get into new training
                
                # ==========================================================
                # I. loss and model update
                # ==========================================================
                micro_step_index = this_step % gradient_accumulation_steps
                # micro_step_index: 1,2,..,GAS-1 => not update the model
                #                   0 => update the model
                
                # ==========================================================
                # I.1. calculate the loss per step/batch
                # ==========================================================
                X_train_batch = item[0].to(device) # (b, vib_mode, seq_len)
                Y_train_batch = item[1].to(device) # (b, seq_len)
                
                loss = wk_ProteinDesigner(
                    Y_train_batch, # use seq_objs channel
                    # 1. text condition via text_embed 
                    text_con_input = X_train_batch, # to be mapped into text_con_embeds
                    # 2. img condition 
                    cond_images = X_train_batch,
                )
                loss = loss/gradient_accumulation_steps # scale it down
                loss.backward()
                # this line does:
                # 1. dloss/dx for all parameters x, 
                # 2. then x.grad += dloss/dx
                # note: the modle is not changed yet
                
                # ==========================================================
                # I.2. update the model at GAS: Gradient Accumulated Step
                # ==========================================================
                if micro_step_index == 0:
                    completed_updating_steps += 1
                    # get lr for this GAS.
                    # NOTE: define lr using GAS
                    # determine and set the learning rate for GAS
                    lr = get_lr_linear_cosine_contant(
                        it=completed_updating_steps,
                        warmup_iters=TrainKeys['warmup_iters'],
                        lr_decay_iters=TrainKeys['lr_decay_iters'],
                        min_lr=TrainKeys['min_lr'],
                        learning_rate=TrainKeys['learning_rate'],
                    ) if TrainKeys['decay_lr'] else TrainKeys['learning_rate']
                    #
                    # assign the lr to param groups
                    #
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    # clip the gradient
                    # accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(
                        wk_ProteinDesigner.parameters(), 
                        TrainKeys['max_grad_norm']
                    )
                    torch.nn.utils.clip_grad_norm_(
                        wk_ProteinDesigner.diffuser_core.unets[0].parameters(), 
                        TrainKeys['max_grad_norm']
                    )
                    # update the model
                    optimizer.step()       # update x using x.grad
                    # reset to the ground
                    # flush the gradients as soon as we can, no need for this memory anymore
                    optimizer.zero_grad(set_to_none=True)
                    
                # ====================================================================
                # II. reporting and recroding
                # ====================================================================
                # II.1. cheap reporting: training loss at some GAS
                # ====================================================================
                # if completed_updating_steps % TrainKeys['report_1_trai_loss_this_GAS']==0:
                if this_step % (
                    TrainKeys['report_1_trai_loss_this_GAS']* \
                    gradient_accumulation_steps
                )==0:
                    this_w_line = f"%d,%d,%d,%f,%f\n" % (
                        epoch, this_step, completed_updating_steps,
                        loss.item()*gradient_accumulation_steps,
                        lr
                    )
                    this_p_line = f"\nepoch: %d, step: %d, GAS: %d, loss/trai: %f, lr: %f\n" % (
                        epoch, this_step, completed_updating_steps,
                        loss.item()*gradient_accumulation_steps,
                        lr
                    )
                    write_one_line_to_file(
                        this_line=this_w_line,
                        file_name=TrainKeys['1_train_loss.log'],
                        mode='a',
                    )
                    Print(this_p_line)
                    
                # 2. medium expensive reporting: vail loss
                if this_step % (
                    TrainKeys['report_2_vali_loss_this_GAS'] * \
                    gradient_accumulation_steps
                )==0:
                    # a. vail loss: on the whole eval_dataloader
                    eval_mean_loss = evaluate_loop_for_loss_of_PD(
                        wk_ProteinDesigner,
                        test_loader,
                        max_batch_to_cover=eval_during_trai_1_max_batch_to_cover, # 5,
                        device=device,
                    )
                    this_w_line = f"%d,%d,%d,%f,%f,%f\n" % (
                        epoch, this_step, completed_updating_steps,
                        loss.item()*gradient_accumulation_steps,
                        eval_mean_loss,
                        lr
                    )
                    this_p_line = f"\nepoch: %d, step: %d, GAS: %d, loss/trai: %f, loss/eval: %f, lr: %f\n" % (
                        epoch, this_step, completed_updating_steps,
                        loss.item()*gradient_accumulation_steps,
                        eval_mean_loss,
                        lr
                    )
                    write_one_line_to_file(
                        this_line=this_w_line,
                        file_name=TrainKeys['2_vali_loss.log'],
                        mode='a',
                    )
                    Print(this_p_line)
                              
                    # b. predict some 
                    # pass
                
                # ====================================================================
                # II.2. save checkpoints: the best, the last and a few
                # ====================================================================
                # 
                if this_step % (
                    TrainKeys['report_3_save_mode_this_GAS'] * \
                    gradient_accumulation_steps
                )==0:
                    # 
                    # ..............................
                    # check if is the current BEST
                    IF_Record_Best = 0
                    if eval_mean_loss < best_val_loss:
                        # prep the state
                        # update key values
                        best_val_loss = eval_mean_loss
                        GAS_at_best_val_loss = completed_updating_steps
                        # make a mark
                        IF_Record_Best = 1
                    # 
                    # ..............................
                    # prepare the model ck
                    checkpoint = {
                        'model': wk_ProteinDesigner.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'completed_updating_steps': completed_updating_steps,
                        'step_num': this_step,
                        'iter_num_at_best_loss': GAS_at_best_val_loss,
                        'best_val_loss': best_val_loss
                    }
                    # 
                    # ..............................
                    # update the best:
                    # NOTE, this saver only has a limited freq
                    if IF_Record_Best==1:
                        Print(f"Update the best ck...")
                        torch.save(
                            checkpoint,
                            os.path.join(
                                TrainKeys['ck_dir_best'],
                                'Best_ckpt.pt'
                            )
                        )
                        # add a note
                        this_w_line = f"%d,%d,%d,%f,%f\n" % (
                            epoch,this_step,completed_updating_steps,
                            loss.item()*gradient_accumulation_steps,
                            best_val_loss
                        )
                        # in a eraseing mode
                        write_one_line_to_file(
                            this_line=this_w_line,
                            file_name=TrainKeys['3_save_model_best.log'],
                            mode='w',
                        )
                    # 
                    # ..............................
                    # REGULAR save
                    # 1. save the last
                    Print(f"Regular save ck at GAS {completed_updating_steps}...\n")
                    torch.save (
                        checkpoint, 
                        os.path.join(
                            TrainKeys['ck_dir_last'],
                            f"Last_ckpt.pt"
                        )
                    )
                    # ++
                    # ..............................
                    # REGULAR save: keep the last N
                    # note: int(file_name.split('_')[2]) should give completed_updating_steps
                    # to save time, we don't use save but just copy
                    # 
                    torch.save(
                        checkpoint, 
                        os.path.join(
                            TrainKeys['ck_dir_last'],
                            f"Keep_ckpt_{completed_updating_steps}_.pt"
                        )
                    )
                    shutil.copyfile(
                        src=os.path.join(
                                TrainKeys['ck_dir_last'],
                                f"Last_ckpt.pt"
                            ), 
                        dst=os.path.join(
                                TrainKeys['ck_dir_last'],
                                f"Keep_ckpt_{completed_updating_steps}_.pt"
                            )
                    )
                    
                    # remove some ckpts if needed
                    keep_only_the_latest_N_files(
                        folder_path=TrainKeys['ck_dir_last'],
                        file_name_starter="Keep",
                        num_to_keep=last_num_ckpt_to_keep,
                    )
                    
                # ====================================================================
                # II.3. expensive reporting: sampling test on only a few
                # ====================================================================
                #
                if this_step % (
                    TrainKeys['report_2_vali_samp_this_GAS'] * \
                    gradient_accumulation_steps
                )==0:
                    ####
                    Print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    Print ("I. SAMPLING IN TEST SET: ")
                    Print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    ####
                    
                    sampling_loop_for_PD_on_DataLoder_w_OF(
                        # 1. on model
                        wk_ProteinDesigner,
                        # 2. on data
                        test_loader,
                        max_batch_to_cover=sampling_max_batch_to_cover, # 2,
                        max_seq_per_batch_to_fold=sampling_max_seq_per_batch_to_fold, # 2,
                        # 3. on sampling and folding
                        cond_scales=sampling_cond_scales, # [7.5],
                        # 4. others
                        device=device,
                        CKeys=CKeys,
                        epoch=str(epoch), # 'SA_test', # 111,
                        GAS=str(completed_updating_steps), # 'SA_test', 
                        # 5. report controler
                        sample_dir=TrainKeys['sampling_dir'], # ModelKeys['model_dir_sample'],
                        sample_prefix=sampling_sample_prefix, # 'Trai', # starter of all saved files
                        # 5.1 on fig of toks 
                        If_Plot_seq_toks=sampling_If_Plot_seq_toks, # (CKeys['SilentRun']==0), #True,
                        IF_showfig=1-CKeys['SilentRun'], # plot or save
                        # 5.2 on basic Prediction package
                        IF_save_prediction_pack = sampling_IF_save_prediction_pack, # 1,
                        # 5.3 on folding prediction via folding tools
                        IF_fold_seq = sampling_IF_fold_seq, # 1, # will produce PBD file
                        IF_show_foldding = 1-CKeys['SilentRun'], 
                        IF_DSSP = sampling_IF_DSSP, # 1, # check SecStr
                    )
                    
                    Print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    Print ("II. SAMPLING with conditioning only: ")
                    Print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    
                    test_X_file_list, test_pdb_file_list, test_fasta_file_list = \
                    sampling_loop_for_PD_on_ConditionList_w_OF(
                        # 1. on model
                        wk_ProteinDesigner,
                        # 2. on input conditioning
                        raw_condition_list=denovo_raw_condition_list, # list of (mode=3, seq_len) np arr
                        noramfac_for_modes=noramfac_for_modes, # DataKeys['Xnormfac'],
                        sample_batch_size=denovo_sample_batch_size, # 256,
                        cond_scales=denovo_cond_scales, # [7.5], # between 5 and 10
                        # 3. on sampling controlers
                        sample_dir=TrainKeys['sampling_dir'], # None,
                        If_Plot_seq_toks=denovo_If_Plot_seq_toks, # True, # generate plot of seq idx
                        IF_showfig=1-CKeys['SilentRun'], # show the plot
                        IF_save_prediction_pack=denovo_IF_save_prediction_pack, # 1, # save seq in fasta
                        IF_fold_seq = denovo_IF_fold_seq, # 1, # fold AA into PDB
                        IF_show_foldding=1-CKeys['SilentRun'], # visualization pd
                        IF_DSSP=denovo_IF_DSSP, # 1, # ana. SecStr
                        # 4. others
                        sample_prefix=denovo_sample_prefix, # "DeNovo",
                        epoch=str(epoch), # 'Test',
                        GAS=str(completed_updating_steps), # 'Test',
                        device=device, # None,
                    )
                    
                    # pass
                    
            # head above
            # if this_step > finished_steps: # get into new training
                # clean the eval tail
                if not wk_ProteinDesigner.training:
                    wk_ProteinDesigner.train()
                    
            else:
                pass 
    
# 
# ==============================================================
# training loop: Protein Predictor
# 
def train_loop_for_PP_on_DataLoder(
    # 0. basic
    train_loader_noshuffle,
    test_loader,
    wk_ProteinPredictor, # model
    optimizer,
    TrainKeys,
    CKeys,
    device,
    # 1. pick-up from previous
    finished_steps = None, # finished_steps_0 # end of the previous training point
    completed_updating_steps = None, # completed_updating_steps_0 # GAS finished previously
    best_val_loss = None, # best_val_loss_0
    GAS_at_best_val_loss = None, # GAS_at_best_val_loss_0
    # 2. freq to check and save
    gradient_accumulation_steps = None, # TrainKeys['gradient_accumulation_steps']
    last_num_ckpt_to_keep = 2,
    
    # 2.1 on eval_loss
    eval_during_trai_1_max_batch_to_cover = 5,
    # 2.2 on sampling on test dataloader
    sampling_max_batch_to_cover = 2,
    sampling_max_seq_per_batch_to_fold = 2,
    sampling_cond_scales = [7.5],
    # 
    sampling_If_Plot_nms_vecs = True,
    sampling_IF_save_prediction_pack = 1,
    # 
    sampling_sample_prefix = 'Trai_Samp',
    # 2.3 on denovo sampling
    denovo_raw_condition_list = None, # raw_condition_list
    noramfac_for_modes = None,
    AA_seq_max_len = None, # PP_ModelKeys['text_seq_len']
    denovo_cond_scales = [7.5], 
    denovo_sample_batch_size = 256,
    denovo_If_print_vecs =True,
    denovo_If_Plot_vecs = True,
    denovo_IF_save_prediction_pack = 1,
    denovo_sample_prefix = 'Trai_Denovo',
    
):
    
    Print(f"Previously, ")
    Print(f" finished_steps_0: {finished_steps}")
    Print(f" completed_updating_steps_0: {completed_updating_steps}")
    Print(f" best_val_loss_0: {best_val_loss}")
    
    
    # prepare
    target_epochs = TrainKeys['num_train_epochs']
    
    wk_ProteinPredictor.train()
    # to make sure: direct talk to unets
    wk_ProteinPredictor.diffuser_core.unets[0].train()
    # clean
    optimizer.zero_grad(set_to_none=True)
    
    this_step = 0 # real mini-batch counter
    for epoch in range(target_epochs):
        for item in train_loader_noshuffle:
            this_step += 1 # start from 1, not 0
            if this_step > finished_steps: # get into new training
                
                # ==========================================================
                # I. loss and model update
                # ==========================================================
                micro_step_index = this_step % gradient_accumulation_steps
                # micro_step_index: 1,2,..,GAS-1 => not update the model
                #                   0 => update the model
                
                # ==========================================================
                # I.1. calculate the loss per step/batch
                # ==========================================================
                X_train_batch = item[0].to(device) # (b, vib_mode, seq_len)
                Y_train_batch = item[1].to(device) # (b, seq_len)
                
                loss = wk_ProteinPredictor(
                    X_train_batch, # (b, vib_mode, seq_len) # use seq_objs channel
                    # 1. text condition via text_embed 
                    text_con_input = Y_train_batch, # to be mapped into text_con_embeds
                    # 2. img condition 
                    cond_images = Y_train_batch,
                )
                loss = loss/gradient_accumulation_steps # scale it down
                loss.backward()
                # this line does:
                # 1. dloss/dx for all parameters x, 
                # 2. then x.grad += dloss/dx
                # note: the modle is not changed yet
                
                # ==========================================================
                # I.2. update the model at GAS: Gradient Accumulated Step
                # ==========================================================
                if micro_step_index == 0:
                    completed_updating_steps += 1
                    # get lr for this GAS.
                    # NOTE: define lr using GAS
                    # determine and set the learning rate for GAS
                    lr = get_lr_linear_cosine_contant(
                        it=completed_updating_steps,
                        warmup_iters=TrainKeys['warmup_iters'],
                        lr_decay_iters=TrainKeys['lr_decay_iters'],
                        min_lr=TrainKeys['min_lr'],
                        learning_rate=TrainKeys['learning_rate'],
                    ) if TrainKeys['decay_lr'] else TrainKeys['learning_rate']
                    #
                    # assign the lr to param groups
                    #
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    # clip the gradient
                    # accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(
                        wk_ProteinPredictor.parameters(), 
                        TrainKeys['max_grad_norm']
                    )
                    torch.nn.utils.clip_grad_norm_(
                        wk_ProteinPredictor.diffuser_core.unets[0].parameters(), 
                        TrainKeys['max_grad_norm']
                    )
                    # update the model
                    optimizer.step()       # update x using x.grad
                    # reset to the ground
                    # flush the gradients as soon as we can, no need for this memory anymore
                    optimizer.zero_grad(set_to_none=True)
                    
                # ====================================================================
                # II. reporting and recroding
                # ====================================================================
                # II.1. cheap reporting: training loss at some GAS
                # ====================================================================
                # if completed_updating_steps % TrainKeys['report_1_trai_loss_this_GAS']==0:
                if this_step % (
                    TrainKeys['report_1_trai_loss_this_GAS']* \
                    gradient_accumulation_steps
                )==0:
                    this_w_line = f"%d,%d,%d,%f,%f\n" % (
                        epoch, this_step, completed_updating_steps,
                        loss.item()*gradient_accumulation_steps,
                        lr
                    )
                    this_p_line = f"\nepoch: %d, step: %d, GAS: %d, loss/trai: %f, lr: %f\n" % (
                        epoch, this_step, completed_updating_steps,
                        loss.item()*gradient_accumulation_steps,
                        lr
                    )
                    write_one_line_to_file(
                        this_line=this_w_line,
                        file_name=TrainKeys['1_train_loss.log'],
                        mode='a',
                    )
                    Print(this_p_line)
                    
                # 2. medium expensive reporting: vail loss
                if this_step % (
                    TrainKeys['report_2_vali_loss_this_GAS'] * \
                    gradient_accumulation_steps
                )==0:
                    # a. vail loss: on the whole eval_dataloader
                    eval_mean_loss = evaluate_loop_for_loss_of_PP(
                        wk_ProteinPredictor,
                        test_loader,
                        max_batch_to_cover=eval_during_trai_1_max_batch_to_cover, # 5,
                        device=device,
                    )
                    this_w_line = f"%d,%d,%d,%f,%f,%f\n" % (
                        epoch, this_step, completed_updating_steps,
                        loss.item()*gradient_accumulation_steps,
                        eval_mean_loss,
                        lr
                    )
                    this_p_line = f"\nepoch: %d, step: %d, GAS: %d, loss/trai: %f, loss/eval: %f, lr: %f\n" % (
                        epoch, this_step, completed_updating_steps,
                        loss.item()*gradient_accumulation_steps,
                        eval_mean_loss,
                        lr
                    )
                    write_one_line_to_file(
                        this_line=this_w_line,
                        file_name=TrainKeys['2_vali_loss.log'],
                        mode='a',
                    )
                    Print(this_p_line)
                              
                    # b. predict some 
                    # pass
                
                # ====================================================================
                # II.2. save checkpoints: the best, the last and a few
                # ====================================================================
                # 
                if this_step % (
                    TrainKeys['report_3_save_mode_this_GAS'] * \
                    gradient_accumulation_steps
                )==0:
                    # 
                    # ..............................
                    # check if is the current BEST
                    IF_Record_Best = 0
                    if eval_mean_loss < best_val_loss:
                        # prep the state
                        # update key values
                        best_val_loss = eval_mean_loss
                        GAS_at_best_val_loss = completed_updating_steps
                        # make a mark
                        IF_Record_Best = 1
                    # 
                    # ..............................
                    # prepare the model ck
                    checkpoint = {
                        'model': wk_ProteinPredictor.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'completed_updating_steps': completed_updating_steps,
                        'step_num': this_step,
                        'iter_num_at_best_loss': GAS_at_best_val_loss,
                        'best_val_loss': best_val_loss
                    }
                    # 
                    # ..............................
                    # update the best:
                    # NOTE, this saver only has a limited freq
                    if IF_Record_Best==1:
                        Print(f"Update the best ck...")
                        torch.save(
                            checkpoint,
                            os.path.join(
                                TrainKeys['ck_dir_best'],
                                'Best_ckpt.pt'
                            )
                        )
                        # add a note
                        this_w_line = f"%d,%d,%d,%f,%f\n" % (
                            epoch,this_step,completed_updating_steps,
                            loss.item()*gradient_accumulation_steps,
                            best_val_loss
                        )
                        # in a eraseing mode
                        write_one_line_to_file(
                            this_line=this_w_line,
                            file_name=TrainKeys['3_save_model_best.log'],
                            mode='w',
                        )
                    # 
                    # ..............................
                    # REGULAR save
                    # 1. save the last
                    Print(f"Regular save ck at GAS {completed_updating_steps}...\n")
                    torch.save (
                        checkpoint, 
                        os.path.join(
                            TrainKeys['ck_dir_last'],
                            f"Last_ckpt.pt"
                        )
                    )
                    # ++
                    # ..............................
                    # REGULAR save: keep the last N
                    # note: int(file_name.split('_')[2]) should give completed_updating_steps
                    # to save time, we don't use save but just copy
                    # 
                    torch.save(
                        checkpoint, 
                        os.path.join(
                            TrainKeys['ck_dir_last'],
                            f"Keep_ckpt_{completed_updating_steps}_.pt"
                        )
                    )
                    shutil.copyfile(
                        src=os.path.join(
                                TrainKeys['ck_dir_last'],
                                f"Last_ckpt.pt"
                            ), 
                        dst=os.path.join(
                                TrainKeys['ck_dir_last'],
                                f"Keep_ckpt_{completed_updating_steps}_.pt"
                            )
                    )
                    
                    # remove some ckpts if needed
                    keep_only_the_latest_N_files(
                        folder_path=TrainKeys['ck_dir_last'],
                        file_name_starter="Keep",
                        num_to_keep=last_num_ckpt_to_keep,
                    )
                    
                # ====================================================================
                # II.3. expensive reporting: sampling test on only a few
                # ====================================================================
                #
                if this_step % (
                    TrainKeys['report_2_vali_samp_this_GAS'] * \
                    gradient_accumulation_steps
                )==0:
                    ####
                    Print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    Print ("I. SAMPLING IN TEST SET: ")
                    Print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    ####
                    
                    csv_file_prefix_on_Test_Loader = \
                    sampling_loop_for_PP_on_DataLoder(
                        # 1. on model
                        wk_ProteinPredictor,
                        # 2. on data
                        test_loader,
                        max_batch_to_cover=sampling_max_batch_to_cover, # 2,
                        max_seq_per_batch_to_fold=sampling_max_seq_per_batch_to_fold, # 2,
                        noramfac_for_modes=noramfac_for_modes, # PP_DataKeys['Xnormfac']
                        # 3. on sampling and folding
                        cond_scales=sampling_cond_scales, # [7.5],
                        # 4. others
                        device=device,
                        CKeys=CKeys,
                        epoch=str(epoch), # 'SA_test', # 111,
                        GAS=str(completed_updating_steps), # 'SA_test', 
                        # 5. report controler
                        sample_dir=TrainKeys['sampling_dir'], # ModelKeys['model_dir_sample'],
                        sample_prefix=sampling_sample_prefix, # 'Trai', # starter of all saved files
                        # 5.1 on fig of nms vecs
                        If_Plot_nms_vecs = sampling_If_Plot_nms_vecs,
                        IF_showfig=1-CKeys['SilentRun'], # plot or save
                        # 5.2 on basic Prediction package: Seq, PR, GTs
                        IF_save_prediction_pack = sampling_IF_save_prediction_pack, # 1,
                    )
                    
                    Print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    Print ("II. SAMPLING with conditioning only: ")
                    Print ("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                    
                    test_X_file_list, test_PR_file_list = \
                    sampling_loop_for_PP_on_ConditionList(
                        # 1. on model
                        wk_ProteinPredictor,
                        # 2. on input conditioning
                        raw_condition_list=denovo_raw_condition_list, # list of (mode=3, seq_len) np arr
                        AA_seq_max_len=AA_seq_max_len,
                        noramfac_for_modes=noramfac_for_modes, # DataKeys['Xnormfac'],
                        sample_batch_size=denovo_sample_batch_size, # 256,
                        cond_scales=denovo_cond_scales, # [7.5], # between 5 and 10
                        # 3. on sampling controlers
                        sample_dir=TrainKeys['sampling_dir'], # None,
                        If_print_vecs=denovo_If_print_vecs, # True
                        If_Plot_vecs=denovo_If_Plot_vecs, # True, # generate plot of NMS vecs
                        IF_showfig=1-CKeys['SilentRun'], # show the plot
                        IF_save_prediction_pack=denovo_IF_save_prediction_pack, # 1, # save seq in fasta
                        # 4. others
                        sample_prefix=denovo_sample_prefix, # "DeNovo",
                        epoch=str(epoch), # 'Test',
                        GAS=str(completed_updating_steps), # 'Test',
                        device=device, # None,
                    )
                    
                    # pass
                    
            # head above
            # if this_step > finished_steps: # get into new training
                # clean the eval tail
                if not wk_ProteinPredictor.training:
                    wk_ProteinPredictor.train()
                    
            else:
                pass 
            
        # return csv_file_prefix_on_Test_Loader
    