# //////////////////////////////////////////////////////
# 0. load in packages
# //////////////////////////////////////////////////////

import math
import numpy as np

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

import esm

# //////////////////////////////////////////////////////////////
# 2. special packages
# //////////////////////////////////////////////////////////////

from VibeGen.imagen_x_unet_pytorch import (
    Unet_OneD, NullUnet, default, Identity
)
#
from VibeGen.imagen_x_imagen_pytorch import (
    ElucidatedImagen_OneD, eval_decorator
)
# 
from VibeGen.imagen_x_trainer_pytorch import (
    ImagenTrainer_OneD
)
#
from VibeGen.UtilityPack import (
    Print, Print_model_params,
    read_mask_from_input,
    keep_only_20AA_channels_in_one_pLM_logits,
    esm_tok_to_idx,
    get_nms_vec_as_arr_list_from_batch_using_mask
)

# //////////////////////////////////////////////////////////////
# 3. local setup parameters: for debug purpose
# //////////////////////////////////////////////////////////////
PD_Init_Level = 1 # for Initialization
PD_Forw_Level = 1 # for forward()
PD_Samp_Level = 1 # for sample()

Local_Debug_Level = 0 # 1
# PD_Init_Level = 0 # for Initialization
# PD_Forw_Level = 0 # for forward()
# PD_Samp_Level = 0 # for sample()

# //////////////////////////////////////////////////////////////
# 4. helper functions
# //////////////////////////////////////////////////////////////
cal_norm_prob = nn.Softmax(dim=2)

# 
# //////////////////////////////////////////////////////////////
# 5. Main class:
# //////////////////////////////////////////////////////////////
# step 1: just a wrap for EImagen
class ProteinDesigner_Base(nn.Module):
    
    def __init__(
        self, 
        # 1. Diffusion core
        unet,
        elucidated=True, # use Elucidated Imagen; others to be implemented 
        # 1.1. on the main obj that passes through the UNet/Diffuser
        seq_obj_max_size=128,  # max AA seq size
        # 1.2 on extra care on text condition; 
        # ohter conditioning is passed in via UNet
        text_embed_input_dim=3, # dim of raw input text_con, 
        # will be mapped into text_con_embed_dim using Linear layer
        text_con_embed_dim=768, # dim of text content embedding
        text_pos_embed_dim=32, # dim for text position embedding
        cond_drop_prob=0.1, 
        # 2. do diffusion/sampling
        timestep = 10,
        # 3. helper for protein Language Model: freeze
        proteinLanguageModel = None,
        
        # ++ 
        # for debug
        CKeys={'Debug_Level':0}
    ):
        # super(ProteinDesigner_Base, self).__init__()
        super().__init__()
        
        Print ("Initialize Diffusion based Protein Designer model...")
        # ++ for debug
        self.CKeys = CKeys
        if self.CKeys['Debug_Level']==PD_Init_Level:
            print ("Debug mode:...\n")
        
        # on the main object
        self.seq_obj_max_size = seq_obj_max_size # max length of the main obj
        self.seq_obj_channels = unet.channels
        
        # ++
        if self.CKeys['Debug_Level']==PD_Init_Level:
            print (f"On main obj:")
            print (f"obj.channels: {unet.channels}")
            print (f"obj.max_len: {self.seq_obj_max_size}")
        
        # on text conditioning
        self.text_embed_input_dim = text_embed_input_dim
        self.text_con_embed_dim = text_con_embed_dim
        text_embed_dim = text_con_embed_dim+text_pos_embed_dim
        assert unet.text_embed_dim==text_embed_dim, "the tot dim of text embedding need to match that of the unet passed in."
        self.text_embed_dim = text_embed_dim
        self.text_cond_drop_prob = cond_drop_prob
        self.text_max_len = unet.max_text_len
        # ++
        if self.CKeys['Debug_Level']==PD_Init_Level:
            if unet.cond_on_text:
                print (f"On text conditioning:")
                print (f"text tot embed dim: {self.text_embed_dim}")
                print (f"text max len: {unet.max_text_len}")
            else:
                print (f"Cond via text: {unet.cond_on_text}")
        # map the text_embed_input_dim to text_embed_input_dim
        self.to_text_con_emb = nn.Linear(
            self.text_embed_input_dim, self.text_con_embed_dim
        )
        # create text positional embedding
        self.to_text_pos_emb = nn.Embedding(
            self.text_max_len+1, text_pos_embed_dim
        )
        self.text_pos_matrix_i = torch.zeros (
            self.text_max_len, dtype=torch.long
        )
        for i in range (self.text_max_len):
            self.text_pos_matrix_i [i]=i +1
            # pos_matrix_i: [1,2,3,...,N,0]
        
        # on img conditioning: passed in via UNet object
        # ++
        if self.CKeys['Debug_Level']==PD_Init_Level:
            if unet.cond_images_channels>0:
                print (f"On Img conditioning:")
                print (f"cond_img_channels: {unet.cond_images_channels}")
            else:
                print (f"Img_condition: False")
        
        # on self_cond: passed in via UNet obejct
        # ++
        if self.CKeys['Debug_Level']==PD_Init_Level:
            print (f"self_cond: {unet.self_cond}")
            
        # on lowres_img cond
        # not used for one unet case
        
        # on diffusion
        self.num_sample_steps = (timestep,)
        
        
        assert elucidated, "Only elucidated diffuser is implemented..."
        self.is_elucidated = elucidated
        # +++
        # try to 
        
        if self.is_elucidated:
            
            self.diffuser_core = ElucidatedImagen_OneD(
                # 1. unets
                unets = (unet,),
                channels = unet.channels, # can be used for unet correcting
                # 2. in-out image size
                image_sizes=(self.seq_obj_max_size,),
                # 3. on text conditioning
                text_encoder_name = None, # TBU: DEFAULT_T5_NAME,
                text_embed_dim = self.text_embed_dim,
                cond_drop_prob = self.text_cond_drop_prob, # 0.1,
                # 4. on cond_img
                auto_normalize_img = True,                  # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
                #     here input image objects should be (0, 1), for Diffusion, need to be (-1,1). So, turn on this 
                # 5. on which to train
                only_train_unet_number = 1, # (1,2)
                # 6. on lowres conditioning
                lowres_noise_schedule = 'linear',
                # 6. on diffusion
                num_sample_steps = self.num_sample_steps,   # number of sampling steps
                sigma_min = 0.002,                          # min noise level
                sigma_max = 160,                            # max noise level
                sigma_data = 0.5,                           # standard deviation of data distribution
                rho = 7,                                    # controls the sampling schedule
                P_mean = -1.2,                              # mean of log-normal distribution from which noise is drawn for training
                P_std = 1.2,                                # standard deviation of log-normal distribution from which noise is drawn for training
                S_churn = 40, # 80,                         # parameters for stochastic sampling - depends on dataset, Table 5 in apper
                S_tmin = 0.05,
                S_tmax = 50,
                S_noise = 1.003,
                # ++
                CKeys = {'Debug_Level':CKeys['Debug_Level']}, # for debug purpose: 0--silence mode

                # =======================================
                # others: just use default values
                random_crop_sizes = None, # default
                resize_mode = 'nearest',  # can be used in UNet upsampling, use default
                temporal_downsample_factor = 1,
                per_sample_random_aug_noise_level = False,  # unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
                dynamic_thresholding = True,
                dynamic_thresholding_percentile = 0.95,     # unsure what this was based on perusal of paper

                # video related: not implemented yet
                resize_cond_video_frames = True, # not act since not video
                # on lowres_image sampling, not use if only one UNet
                lowres_sample_noise_level = 0.2,            # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
            )
        
        # default to device of diffuser_core, a EImagen, passed in
        
        self.device = next(self.diffuser_core.parameters()).device
        self.to(self.device)
        
        # ==================================================
        # on pretrained Protein Language Models
        self.pLM_Name = proteinLanguageModel
        pLM_Model, esm_alphabet, esm_layer, len_toks\
        = self.prepare_protein_LM(
            freeze_pLM=True,
        )
        self.pLM = pLM_Model
        self.pLM_alphabet = esm_alphabet
        self.pLM_esm_layer = esm_layer
        self.pLM_len_toks = len_toks
        
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # helpers
    def turn_on_train_mode(
        self,
    ):
        # from the top level
        self.train()
        # make sure 
        self.diffuser_core.unets[0].train()
        
    def turn_on_eval_mode(
        self,
    ):
        # from the top level
        self.eval()
        # make sure 
        self.diffuser_core.unets[0].eval()
    
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
    def prepare_protein_LM(
        self,
        freeze_pLM=True,
    ):
        pLM_Model_Name = self.pLM_Name
        device = self.device
        
        if self.CKeys['Debug_Level']==PD_Forw_Level:
            print (f"Get pLM: {pLM_Model_Name}")
            
        # ++ for pLM
        if pLM_Model_Name is None:
            # may add a plain tokenizer here if needed
            pLM_Model=None
            esm_alphabet=None
            esm_layer=0
            len_toks=0

        elif pLM_Model_Name=='esm2_t33_650M_UR50D':
            # dim: 1280
            esm_layer=33
            pLM_Model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            len_toks=len(esm_alphabet.all_toks)
            pLM_Model.eval()
            pLM_Model. to(device)

        elif pLM_Model_Name=='esm2_t36_3B_UR50D':
            # dim: 2560
            esm_layer=36
            pLM_Model, esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            len_toks=len(esm_alphabet.all_toks)
            pLM_Model.eval()
            pLM_Model. to(device)

        elif pLM_Model_Name=='esm2_t30_150M_UR50D':
            # dim: 640
            esm_layer=30
            pLM_Model, esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            len_toks=len(esm_alphabet.all_toks)
            pLM_Model.eval()
            pLM_Model. to(device)

        elif pLM_Model_Name=='esm2_t12_35M_UR50D':
            # dim: 480
            esm_layer=12
            pLM_Model, esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            len_toks=len(esm_alphabet.all_toks)
            pLM_Model.eval()
            pLM_Model. to(device)

        else:
            Print("pLM model is missing...")
            
        # freeze the pretrained pLM
        
        if freeze_pLM and not (pLM_Model_Name is None):
            
            for param in pLM_Model.parameters():
                param.requires_grad = False
        
        return pLM_Model, esm_alphabet, esm_layer, len_toks
    
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
    @torch.no_grad()
    def map_seq_tok_to_seq_channel_w_pLM(self, seq_objs):
        # ref: https://github.com/Bo-Ni/ProteinMechanicsDiffusionDesign_pLDM/blob/main/PD_pLMProbXDiff/TrainerPack.py
        # assume self.pLM is not trivial
        esm_resu = self.pLM(
            seq_objs,
            repr_layers=[self.pLM_esm_layer],
            return_contacts=False,
        )
        # include logits and representations
        # check we need logits or hidden states
        if self.diffuser_core.channels==self.pLM_len_toks:
            # choose logits
            norm_logits = cal_norm_prob(esm_resu['logits'])
            # (batch, seq_len, num_toks)
            norm_logits = rearrange(norm_logits, 'b s t -> b t s') 
            # (batch, num_toks, seq_len)
            
            return norm_logits
        else:
            # choose hidden state
            # need to map them btw 0 and 1 for Diffuser
            Print (f"esm hidden state as main obj is not Normalized yet.")
            # break
            # to use, try the following
            hidden_state = esm_resu['representations'][self.pLM_esm_layer] 
            # (batch, seq_len, dim_hidden)
            hidden_state = rearrange(hidden_state, 'b s h -> b h s') 
            # (batch, dim_hidden, seq_len)
            
            return hidden_state
            # (batch, dim_hidden, seq_len)
    
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
    def forward (
        self,
        seq_objs, # (batch, seq_len)
        seq_objs_embeds = None, # (batch, emb_dim, seq_len)
        # 1. text condition via text_embed
        text_con_input = None,
        text_con_embeds = None,
        # 2. img condition 
        cond_images = None,
        # 3. others
        unet_number = 1,
        # others
        **kwargs
    ):
        '''
        goal: calculate loss. pLM is not called within here
        '''

        # ++
        if self.CKeys['Debug_Level']==PD_Forw_Level:
            print (f"In ProteinDesigner.forward() ...")
            
        # on seq_objs
        # ..............................................
        
        # 1. check which input of seq_objs is used
        assert not ((seq_objs is not None) and (seq_objs_embeds is not None)), \
        "Cannot provide both seq_objs and seq_objs_embeds"
        
        # 2. map seq_objs into seq_objs_embeds using pLM
        if seq_objs is not None:
            
            assert (self.pLM_Name is not None), \
            "Input seq_objs needs a non-trivial pLM..."
            
            seq_objs_embeds = self.map_seq_tok_to_seq_channel_w_pLM(
                seq_objs,
            )
            # ++
            if self.CKeys['Debug_Level']==PD_Forw_Level:
                print (f"pLM produce seq_objs_embeds.shape: {seq_objs_embeds.shape}")
                print (f"check normalization:")
                print (f"sum: {torch.sum(seq_objs_embeds[0,1,:])}")
            
        
        # text_con channel
        # ..............................................
        
        # 0. map low-dim text_con to full-dim text_con_embeds
        if (text_con_input!=None) & (text_con_embeds==None):
            # ++
            if self.CKeys['Debug_Level']==PD_Forw_Level:
                print (f"Provided text_con_input.shape: {text_con_input.shape}") 
                # (b, mode, text_len)
            
            text_con_input = rearrange(text_con_input, 'b m n -> b n m') 
            # (b, text_len, mode)
            # ++
            if self.CKeys['Debug_Level']==PD_Forw_Level:
                print (f"Change as text_emb style, ")
                print (f"text_con_input.shape: {text_con_input.shape}") 
            
            text_con_embeds = self.to_text_con_emb(text_con_input)
            # ++
            if self.CKeys['Debug_Level']==PD_Forw_Level:
                print (f"Map into text_con_emb dim: .to_text_con_emb(text_con_input)")
                print (f"->text_con_embeds.shape: {text_con_embeds.shape}")
            
        # 1. complete text_embeds by adding position part
        if text_con_embeds != None:
        # if text_con_input != None:
            # 1.1 check the length of the text_embeds
            assert text_con_embeds.shape[1]<=self.text_max_len, "text_embed length is larger than defined text_max_len"
            # check the shape of the input text_con_embeds: (batch, text_len, embeds)
            if text_con_embeds.shape[1]<self.text_max_len:
                # need to change the length: assume text_emb padding is zero
                text_con_embeds_fixed = torch.zeros(
                    text_con_embeds.shape[0],
                    self.text_max_len,
                    text_con_embeds.shape[2],
                ).to(self.diffuser_core.device)
                text_con_embeds_fixed[:,:text_con_embeds.shape[1],:]=text_con_embeds
                text_con_embeds = text_con_embeds_fixed
            # 
            # 1.2 provide positional embedding: text_pos_matrix_i
            text_pos_mat_i_ = self.text_pos_matrix_i.repeat(
                text_con_embeds.shape[0], 1
            ).to(self.diffuser_core.device) # (batch, text_len)
            text_pos_embeds = self.to_text_pos_emb(text_pos_mat_i_) 
            # (batch, text_len, text_pos_emb_dim)
            # 1.3 merge con and pos
            text_embeds = torch.cat( (text_con_embeds, text_pos_embeds),2 )
            # ++
            if self.CKeys['Debug_Level']==PD_Forw_Level:
                print (f"text_pos_embeds.shape: {text_pos_embeds.shape}")
                print (f"text_con_embeds.shape: {text_con_embeds.shape}")
                print (f"After merge, text_embeds.shape: {text_embeds.shape}")
            
        # 2. call EImagen
        # ..............................................
        # ++
        if self.CKeys['Debug_Level']==PD_Forw_Level:
            if cond_images!=None:
                print (f"cond_images.shape: {cond_images.shape}")
            
        loss = self.diffuser_core(
            seq_objs_embeds,
            text_embeds = text_embeds,
            cond_images = cond_images,
            unet_number = unet_number,
            **kwargs
        )
            
        
        return loss
    
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
    @torch.no_grad()
    @eval_decorator
    def sample (
        self,
        # 1. on text condition
        texts: List[str] = None, # not used yet
        text_masks = None,
        # 1.1 pick one of the following, text_con_input>text_embeds
        text_con_input = None, # main channel for text
        text_embeds = None,    # backdoor channel: for test only
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
        cond_scale = 7.5,   
        # Researcher Netruk44 have reported 5-10 to be optimal, but anything greater than 10 to break.
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
        '''
        Goal: produce sample output based on
        1. text_embeds
        2. cond_img
        '''
        # ++
        if self.CKeys['Debug_Level']==PD_Forw_Level:
            print (f"In ProteinDesigner.sample() ...")
        
        device = default(device, self.device)
        
        # 1. on text_emb channel:
        # ................................................................
        # target: text_embeds
        # shape: (batch, text_len, text_emb_dim)
        assert not ((text_con_input!=None) & (text_embeds!=None)), \
        "text_embeds and text_con_input cannot be provided at the same time"
        if (text_con_input!=None):
            
            # 1.1. reshape
            text_con_input = rearrange(text_con_input, 'b m n -> b n m') 
            # (batch, seq_len, mode)
            
            # 1.2. expand to text_con_embeds
            text_con_embeds = self.to_text_con_emb(text_con_input)
            # (batch, seq_len, text_con_embeds_dim)
            
            # ++
            if self.CKeys['Debug_Level']==PD_Samp_Level:
                print (f"text_con_embeds.shape: {text_con_embeds.shape}")
            
        # 1.3. add int text_con_pos part
        if text_con_embeds != None:
        # if text_con_input != None:
            # 1.3.1 check the length of the text_embeds
            assert text_con_embeds.shape[1]<=self.text_max_len, "text_embed length is larger than defined text_max_len"
            # check the shape of the input text_con_embeds: (batch, text_len, embeds)
            if text_con_embeds.shape[1]<self.text_max_len:
                # need to change the length: assume text_emb padding is zero
                text_con_embeds_fixed = torch.zeros(
                    text_con_embeds.shape[0],
                    self.text_max_len,
                    text_con_embeds.shape[2],
                ).to(self.diffuser_core.device)
                text_con_embeds_fixed[:,:text_con_embeds.shape[1],:]=text_con_embeds
                text_con_embeds = text_con_embeds_fixed
            # 
            # 1.3.2 provide positional embedding: text_pos_matrix_i
            text_pos_mat_i_ = self.text_pos_matrix_i.repeat(
                text_con_embeds.shape[0], 1
            ).to(device) # (batch, text_len)
            text_pos_embeds = self.to_text_pos_emb(text_pos_mat_i_) 
            # (batch, text_len, text_pos_emb_dim)
            # 1.3 merge con and pos
            text_embeds = torch.cat( (text_con_embeds, text_pos_embeds),2 )
            # ++
            if self.CKeys['Debug_Level']==PD_Samp_Level:
                print (f"text_pos_embeds.shape: {text_pos_embeds.shape}")
                print (f"text_con_embeds.shape: {text_con_embeds.shape}")
                print (f"After merge, text_embeds.shape: {text_embeds.shape}")
        
        # 2. on cond_img channel:
        # ................................................................
        
        
        # pass into the diffuser:EImagen
        output = self.diffuser_core.sample(
            # 1. on text condition
            texts  = texts,
            text_masks = text_masks,
            text_embeds = text_embeds,
            # 2. on condition images
            cond_images = cond_images,
            cond_video_frames = cond_video_frames,
            post_cond_video_frames = post_cond_video_frames,
            # 3. inpaint images
            inpaint_videos = inpaint_videos,
            inpaint_images = inpaint_images,
            inpaint_masks = inpaint_masks,
            inpaint_resample_times = inpaint_resample_times,
            # 
            init_images = init_images,
            skip_steps = skip_steps,
            sigma_min = sigma_min,
            sigma_max = sigma_max,
            video_frames = video_frames,
            batch_size = batch_size,
            cond_scale = cond_scale,
            lowres_sample_noise_level = lowres_sample_noise_level,
            start_at_unet_number = start_at_unet_number,
            start_image_or_video = start_image_or_video,
            stop_at_unet_number = stop_at_unet_number,
            return_all_unet_outputs = return_all_unet_outputs,
            return_pil_images = return_pil_images,
            use_tqdm = use_tqdm,
            use_one_unet_in_gpu = use_one_unet_in_gpu,
            device = device,
        )
        # output: (batch, )
        
        
        return output
    # 
    # ================================================================
    # input: X_batch being padded and normalized
    # will get shape-mask from X_batch if not provided
    # 
    @torch.no_grad()
    @eval_decorator
    def sample_to_pLM_idx_seq (
        self,
        # added ones
        common_AA_only=True,
        mask_from_Y=None,
        # for self.sample method
        # ==========================
        # 1. on text condition
        texts: List[str] = None, # not used yet
        text_masks = None,
        # 1.1 pick one of the following, text_con_input>text_embeds
        text_con_input = None, # main channel for text
        text_embeds = None,    # backdoor channel: for test only
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
        cond_scale = 7.5,   
        # Researcher Netruk44 have reported 5-10 to be optimal, but anything greater than 10 to break.
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
        # 1. call the diffusion core
        # assume input includes
        # text_con_input: (b, .text_embed_input_dim, .text_max_len)
        # cond_images:    (b, .diffuser_core.unets[0].cond_images_channels, .diffuser_core.image_sizes[0])
        output_diffuser = self.sample (
            # 1. on text condition
            texts,
            text_masks = text_masks,
            # 1.1 pick one of the following, text_con_input>text_embeds
            text_con_input = text_con_input, # main channel for text
            text_embeds = text_embeds,    # backdoor channel: for test only
            # 2. on condition images
            cond_images = cond_images,
            cond_video_frames = cond_video_frames,
            post_cond_video_frames = post_cond_video_frames,
            # 3. inpaint images
            inpaint_videos = inpaint_videos,
            inpaint_images = inpaint_images,
            inpaint_masks = inpaint_masks,
            inpaint_resample_times = inpaint_resample_times,
            # 
            init_images = init_images,
            skip_steps = skip_steps,
            sigma_min = sigma_min,
            sigma_max = sigma_max,
            video_frames = video_frames,
            batch_size = batch_size,
            cond_scale = cond_scale,   
            # Researcher Netruk44 have reported 5-10 to be optimal, but anything greater than 10 to break.
            lowres_sample_noise_level = lowres_sample_noise_level,
            start_at_unet_number = start_at_unet_number,
            start_image_or_video = start_image_or_video,
            stop_at_unet_number = stop_at_unet_number,
            return_all_unet_outputs = return_all_unet_outputs,
            return_pil_images = return_pil_images,
            use_tqdm = use_tqdm,
            use_one_unet_in_gpu = use_one_unet_in_gpu,
            device = device,
        )
        
        # 2. tanslate embed into toks using pLM
        if self.seq_obj_channels==self.pLM_len_toks:
            # prob is predicted by Diffuser
            logits = rearrange(output_diffuser, 'b c l -> b l c')
            
        else:
            Print(f"a non-prob case to be implemented...")
            # ToDo: hidden state --> logits using pLM
            return
        
        # # may augment the results by only payying attension to 
        # # SOME of the c=33 tokens
        # if common_AA_only:
        #     logits = keep_only_20AA_channels_in_pLM_logits(
        #         logits,
        #     ) # (b, l, c=33)
        # tokens = logits.max(2).indices # (b, l)
        
        # 3. read mask from the input conditions
        if mask_from_Y is None:
            # first try from cond_images, one channel is enough
            # (b, c, image_size)
            if self.diffuser_core.unets[0].has_cond_image:
                X_cond = cond_images[:,0,:] # (b, seq_len)

            elif self.diffuser_core.unets[0].cond_on_text:
                # secondary, try text channel
                X_cond = text_con_input[:,0,:] # (b, text_len)

            else:
                Print (f"No conditioning is found")

            # 4. create mask
            result_mask = read_mask_from_input(
                tokenized_data=X_cond,
                mask_value=0,
                #
                seq_data=None,
                max_seq_length=None,
            ) # (b, seq_len)
        else:
            result_mask=mask_from_Y # (b, seq_len)
        
        # 5. translate tokens into AA
        tokens_list, seq_list = self.decode_many_esm_logits_w_mask(
            logits,
            result_mask,
            common_AA_only=common_AA_only,
        )
        
        
        
        return tokens_list, seq_list, result_mask
    
    def decode_one_esm_logits_w_mask(
        self,
        #
        this_logits, # (seq_len, channels=33)
        this_mask=None,   # (seq_len)
        #
        common_AA_only=True,
    ):
        if this_mask is None:
            # No mask is provided
            # trust the result is reasonable by itself
            this_token = this_logits.max(1).indices
            # (b, seq_len)
        else:
            # mask is provided
            # relying on mask to know where bos, eos are
            if common_AA_only:
                # this must be used with mask
                this_logits = keep_only_20AA_channels_in_one_pLM_logits(
                    this_logits
                )
            #++
            if Local_Debug_Level==1:
                print (f"this_logits.shape: {this_logits.shape}")
                
            this_token = this_logits.max(1).indices
            # (seq_len)
            # apply mask
            this_token = this_token[this_mask==True] # only work for 1D tensor as it will become flat
            
        # translate back into string seq
        this_seq = []
        for ii in range(len(this_token)):
            this_seq.append(
                self.pLM_alphabet.get_tok(
                    this_token[ii]
                )
            )
        this_seq_string = "".join(this_seq)
        
        return this_token, this_seq_string
    
    # this one must be used with batch_mask
    def decode_many_esm_logits_w_mask(
        self,
        #
        batch_logits,
        batch_mask,
        #
        common_AA_only=True,
    ):
        if batch_mask is None:
            # make a fake list
            batch_mask = [None for ii in range(len(batch_logits))]
            
        batch_tokens = []
        batch_seq = []
        for jj in range(len(batch_logits)):
            this_tokens, this_seq_string = self.decode_one_esm_logits_w_mask(
                
                batch_logits[jj],
                this_mask=batch_mask[jj],
                #
                common_AA_only=common_AA_only,
            )
            batch_tokens.append(this_tokens)
            batch_seq.append(this_seq_string)
        
        return batch_tokens, batch_seq
    
    # get shape-mask based on toks in Y
    def read_mask_from_seq_toks_using_pLM(
        self,
        #
        batch_Y, # (batch, seq_len)
    ):
        # in ESM, looks like 0 xxx 2, 1,1,...1
       
        
        mask_from_Y = torch.logical_and(
            batch_Y!=esm_tok_to_idx['<cls>'],
            batch_Y!=esm_tok_to_idx['<eos>']
        )
        mask_from_Y = torch.logical_and(
            mask_from_Y,
            batch_Y!=esm_tok_to_idx['<pad>']
        )
        
        return mask_from_Y
    
# 
# //////////////////////////////////////////////////////////////
# 6. Main class: Protein predictor
# //////////////////////////////////////////////////////////////
# 
class ProteinPredictor_Base(nn.Module):
    
    def __init__(
        self, 
        # 1. Diffusion core
        unet,
        elucidated=True, # use Elucidated Imagen; others to be implemented 
        # 1.1. on the main obj that passes through the UNet/Diffuser
        seq_obj_max_size=128,  # max AA seq size
        # 1.2 on extra care on text condition; 
        # ohter conditioning is passed in via UNet
        text_embed_input_dim=3, # dim of raw input text_con, 
        # will be mapped into text_con_embed_dim using Linear layer
        text_con_embed_dim=768, # dim of text content embedding
        text_pos_embed_dim=32, # dim for text position embedding
        cond_drop_prob=0.1, 
        # 2. do diffusion/sampling
        timestep = 10,
        # 3. helper for protein Language Model: freeze
        proteinLanguageModel = None,
        
        # ++ 
        # for debug
        CKeys={'Debug_Level':0}
    ):
        # super(ProteinDesigner_Base, self).__init__()
        super().__init__()
        
        Print ("Initialize Diffusion based Protein Designer model...")
        # ++ for debug
        self.CKeys = CKeys
        if self.CKeys['Debug_Level']==PD_Init_Level:
            print ("Debug mode:...\n")
        
        # on the main object
        self.seq_obj_max_size = seq_obj_max_size # max length of the main obj
        self.seq_obj_channels = unet.channels
        
        # ++
        if self.CKeys['Debug_Level']==PD_Init_Level:
            print (f"On main obj:")
            print (f"obj.channels: {unet.channels}")
            print (f"obj.max_len: {self.seq_obj_max_size}")
        
        # on text conditioning
        self.text_embed_input_dim = text_embed_input_dim
        self.text_con_embed_dim = text_con_embed_dim
        text_embed_dim = text_con_embed_dim+text_pos_embed_dim
        assert unet.text_embed_dim==text_embed_dim, "the tot dim of text embedding need to match that of the unet passed in."
        self.text_embed_dim = text_embed_dim
        self.text_cond_drop_prob = cond_drop_prob
        self.text_max_len = unet.max_text_len
        # ++
        if self.CKeys['Debug_Level']==PD_Init_Level:
            if unet.cond_on_text:
                print (f"On text conditioning:")
                print (f"text tot embed dim: {self.text_embed_dim}")
                print (f"text max len: {unet.max_text_len}")
            else:
                print (f"Cond via text: {unet.cond_on_text}")
        
        if text_pos_embed_dim == 0:
            # for PREDICTOR, we use pLM to get full text_embed
            self.to_text_con_emb = Identity()
            self.to_text_pos_emb = None
            self.text_pos_matrix_i = None
        else:
            # for DESIGNER, 
            # map the text_embed_input_dim to text_embed_input_dim
            self.to_text_con_emb = nn.Linear(
                self.text_embed_input_dim, self.text_con_embed_dim
            )
            # create text positional embedding
            self.to_text_pos_emb = nn.Embedding(
                self.text_max_len+1, text_pos_embed_dim
            )
            self.text_pos_matrix_i = torch.zeros (
                self.text_max_len, dtype=torch.long
            )
            for i in range (self.text_max_len):
                self.text_pos_matrix_i [i]=i +1
                # pos_matrix_i: [1,2,3,...,N,0]
        
        # on img conditioning: passed in via UNet object
        # ++
        if self.CKeys['Debug_Level']==PD_Init_Level:
            if unet.cond_images_channels>0:
                print (f"On Img conditioning:")
                print (f"cond_img_channels: {unet.cond_images_channels}")
            else:
                print (f"Img_condition: False")
        
        # on self_cond: passed in via UNet obejct
        # ++
        if self.CKeys['Debug_Level']==PD_Init_Level:
            print (f"self_cond: {unet.self_cond}")
            
        # on lowres_img cond
        # not used for one unet case
        
        # on diffusion
        self.num_sample_steps = (timestep,)
        
        
        assert elucidated, "Only elucidated diffuser is implemented..."
        self.is_elucidated = elucidated
        # +++
        # try to 
        
        if self.is_elucidated:
            
            self.diffuser_core = ElucidatedImagen_OneD(
                # 1. unets
                unets = (unet,),
                channels = unet.channels, # can be used for unet correcting
                # 2. in-out image size
                image_sizes=(self.seq_obj_max_size,),
                # 3. on text conditioning
                text_encoder_name = None, # TBU: DEFAULT_T5_NAME,
                text_embed_dim = self.text_embed_dim,
                cond_drop_prob = self.text_cond_drop_prob, # 0.1,
                # 4. on cond_img
                auto_normalize_img = True,                  # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
                #     here input image objects should be (0, 1), for Diffusion, need to be (-1,1). So, turn on this 
                # 5. on which to train
                only_train_unet_number = 1, # (1,2)
                # 6. on lowres conditioning
                lowres_noise_schedule = 'linear',
                # 6. on diffusion
                num_sample_steps = self.num_sample_steps,   # number of sampling steps
                sigma_min = 0.002,                          # min noise level
                sigma_max = 160,                            # max noise level
                sigma_data = 0.5,                           # standard deviation of data distribution
                rho = 7,                                    # controls the sampling schedule
                P_mean = -1.2,                              # mean of log-normal distribution from which noise is drawn for training
                P_std = 1.2,                                # standard deviation of log-normal distribution from which noise is drawn for training
                S_churn = 40, # 80,                         # parameters for stochastic sampling - depends on dataset, Table 5 in apper
                S_tmin = 0.05,
                S_tmax = 50,
                S_noise = 1.003,
                # ++
                CKeys = {'Debug_Level':CKeys['Debug_Level']}, # for debug purpose: 0--silence mode

                # =======================================
                # others: just use default values
                random_crop_sizes = None, # default
                resize_mode = 'nearest',  # can be used in UNet upsampling, use default
                temporal_downsample_factor = 1,
                per_sample_random_aug_noise_level = False,  # unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
                dynamic_thresholding = True,
                dynamic_thresholding_percentile = 0.95,     # unsure what this was based on perusal of paper

                # video related: not implemented yet
                resize_cond_video_frames = True, # not act since not video
                # on lowres_image sampling, not use if only one UNet
                lowres_sample_noise_level = 0.2,            # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
            )
        
        # default to device of diffuser_core, a EImagen, passed in
        
        self.device = next(self.diffuser_core.parameters()).device
        self.to(self.device)
        
        # ==================================================
        # on pretrained Protein Language Models
        self.pLM_Name = proteinLanguageModel
        pLM_Model, esm_alphabet, esm_layer, len_toks\
        = self.prepare_protein_LM(
            freeze_pLM=True,
        )
        self.pLM = pLM_Model
        self.pLM_alphabet = esm_alphabet
        self.pLM_esm_layer = esm_layer
        self.pLM_len_toks = len_toks
        
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # helpers
    def turn_on_train_mode(
        self,
    ):
        # from the top level
        self.train()
        # make sure 
        self.diffuser_core.unets[0].train()
        
    def turn_on_eval_mode(
        self,
    ):
        # from the top level
        self.eval()
        # make sure 
        self.diffuser_core.unets[0].eval()
    
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
    def prepare_protein_LM(
        self,
        freeze_pLM=True,
    ):
        pLM_Model_Name = self.pLM_Name
        device = self.device
        
        if self.CKeys['Debug_Level']==PD_Forw_Level:
            print (f"Get pLM: {pLM_Model_Name}")
            
        # ++ for pLM
        if pLM_Model_Name is None:
            # may add a plain tokenizer here if needed
            pLM_Model=None
            esm_alphabet=None
            esm_layer=0
            len_toks=0

        elif pLM_Model_Name=='esm2_t33_650M_UR50D':
            # dim: 1280
            esm_layer=33
            pLM_Model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            len_toks=len(esm_alphabet.all_toks)
            pLM_Model.eval()
            pLM_Model. to(device)

        elif pLM_Model_Name=='esm2_t36_3B_UR50D':
            # dim: 2560
            esm_layer=36
            pLM_Model, esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            len_toks=len(esm_alphabet.all_toks)
            pLM_Model.eval()
            pLM_Model. to(device)

        elif pLM_Model_Name=='esm2_t30_150M_UR50D':
            # dim: 640
            esm_layer=30
            pLM_Model, esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            len_toks=len(esm_alphabet.all_toks)
            pLM_Model.eval()
            pLM_Model. to(device)

        elif pLM_Model_Name=='esm2_t12_35M_UR50D':
            # dim: 480
            esm_layer=12
            pLM_Model, esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            len_toks=len(esm_alphabet.all_toks)
            pLM_Model.eval()
            pLM_Model. to(device)

        else:
            Print("pLM model is missing...")
            
        # freeze the pretrained pLM
        
        if freeze_pLM and not (pLM_Model_Name is None):
            
            for param in pLM_Model.parameters():
                param.requires_grad = False
        
        return pLM_Model, esm_alphabet, esm_layer, len_toks
    
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
    @torch.no_grad()
    def map_seq_tok_to_logits_and_hidden_state_w_pLM(self, seq_objs):
        # ref: https://github.com/Bo-Ni/ProteinMechanicsDiffusionDesign_pLDM/blob/main/PD_pLMProbXDiff/TrainerPack.py
        # assume self.pLM is not trivial
        esm_resu = self.pLM(
            seq_objs,
            repr_layers=[self.pLM_esm_layer],
            return_contacts=False,
        )
        # include logits and representations
        # 1. get logits
        norm_logits = cal_norm_prob(esm_resu['logits'])
        # norm_logits = esm_resu['logits']
        # (batch, seq_len, num_toks)
        norm_logits = rearrange(norm_logits, 'b s t -> b t s') 
        # (batch, num_toks, seq_len)
        
        # 2. get hidden state
        hidden_state = esm_resu['representations'][self.pLM_esm_layer] 
        # (batch, seq_len, dim_hidden)
        hidden_state = rearrange(hidden_state, 'b s h -> b h s') 
        # (batch, dim_hidden, seq_len)

        return hidden_state, norm_logits
    
    
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # for Protein Predictor
    def forward (
        self,
        seq_objs, # (batch, n_mode, seq_len)
        seq_objs_embeds = None, # (batch, emb_dim, seq_len), not used for Predictor yet
        # 1. text condition via text_embed
        text_con_input = None,
        text_con_embeds = None,
        # 2. img condition 
        cond_images = None,
        # 3. others
        unet_number = 1,
        # others
        **kwargs
    ):
        '''
        goal: calculate loss. 
        '''

        # ++
        if self.CKeys['Debug_Level']==PD_Forw_Level:
            print (f"In ProteinPredictor.forward() ...")
            
        # on objs: the NMS vectors
        # ..............................................
        
        # 1. check which input of seq_objs is used
        assert not ((seq_objs is not None) and (seq_objs_embeds is not None)), \
        "Cannot provide both seq_objs and seq_objs_embeds"
        
        # 2. map objs to objs_embeds if needed. Skipped here as we use plain vetors
        if seq_objs is not None:
            # if needed, here add the block to translate seq_objs into seq_objs_embeds
            seq_objs_embeds = seq_objs
        else:
            # here, assume seq_objs_embeds is provided. Do nothing
            pass
        batch_size = seq_objs_embeds.shape[0]
        
        # text_con channel: AA seq (b, seq_len)
        # ..............................................
        
        # 0. choose which type of text_input
        # After this part, only text_con_embeds is passed on
        if (text_con_input!=None) & (text_con_embeds==None):
            # here, text_con_input is activated
            # text_con_input is AA seq in (b, seq_len)
            # use pLM to get (b, seq_len, hidden_space_dim)
            assert (self.pLM_Name is not None), \
            "Input seq_objs needs a non-trivial pLM..."
            
            text_con_input, seq_logits =\
            self.map_seq_tok_to_logits_and_hidden_state_w_pLM(
                text_con_input,
            ) 
            # hidden state: (batch, dim_hidden, seq_len)
            # normalized logits: (batch, dim_logits, seq_len)
            
            text_con_input = rearrange(text_con_input, 'b m n -> b n m') 
            # (b, text_len, dim_hidden)
            
            # .to_text_con_emb is Identiy() here for Predictor
            text_con_embeds = self.to_text_con_emb(text_con_input)
            # (b, text_len, dim_hidden)
            
            # ++
            if self.CKeys['Debug_Level']==PD_Forw_Level:
                print (f"pLM produce text_con_embeds.shape: {text_con_embeds.shape}")
                print (f"check normalization:")
                print (f"sum: {torch.sum(text_con_embeds[0,1,:])}")
                
        
        # 1. complete text_embeds by adding position part
        # after this part, only text_embeds is passed on
        if text_con_embeds != None:
            # if text_con_input != None:
            # 1.1 check the length of the text_embeds
            assert text_con_embeds.shape[1]<=self.text_max_len, "text_embed length is larger than defined text_max_len"
            # check the shape of the input text_con_embeds: (batch, text_len, embeds)
            if text_con_embeds.shape[1]<self.text_max_len:
                # need to change the length: assume text_emb padding is zero
                text_con_embeds_fixed = torch.zeros(
                    text_con_embeds.shape[0],
                    self.text_max_len,
                    text_con_embeds.shape[2],
                ).to(self.diffuser_core.device)
                text_con_embeds_fixed[:,:text_con_embeds.shape[1],:]=text_con_embeds
                text_con_embeds = text_con_embeds_fixed
            # 
            # 1.2 provide positional embedding: text_pos_matrix_i if NEEDED
            if self.text_pos_matrix_i is None:
                # for Predictor, pLM is already used.
                text_pos_embeds = torch.zeros(
                    batch_size,
                    self.text_max_len,
                    0,
                ).to(self.diffuser_core.device)
                # (b, text_len, 0), only a trivial one
            else:
                # kept for Designer
                text_pos_mat_i_ = self.text_pos_matrix_i.repeat(
                    text_con_embeds.shape[0], 1
                ).to(self.diffuser_core.device) # (batch, text_len)
                text_pos_embeds = self.to_text_pos_emb(text_pos_mat_i_) 
                # (batch, text_len, text_pos_emb_dim)
                
            # 1.3 merge con and pos
            text_embeds = torch.cat( (text_con_embeds, text_pos_embeds),2 )
            # ++
            if self.CKeys['Debug_Level']==PD_Forw_Level:
                print (f"text_pos_embeds.shape: {text_pos_embeds.shape}")
                print (f"text_con_embeds.shape: {text_con_embeds.shape}")
                print (f"After merge, text_embeds.shape: {text_embeds.shape}")
                
        # cond_images channel: AA seq (b, seq_len)
        # ..............................................
        if cond_images.dim() == 2: # for AA as the input
            cond_images_embeds = seq_logits
            # (b, dim_logits, seq_len)
            # ++
            if self.CKeys['Debug_Level']==PD_Forw_Level:
                if cond_images!=None:
                    print (f"cond_images_embeds.shape: {cond_images_embeds.shape}")
        
        
        # 2. call EImagen
        # ..............................................
            
        loss = self.diffuser_core(
            seq_objs_embeds,
            text_embeds = text_embeds,
            cond_images = cond_images_embeds,
            unet_number = unet_number,
            **kwargs
        )
            
        
        return loss
    
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
    @torch.no_grad()
    @eval_decorator
    def sample (
        self,
        # 1. on text condition
        texts: List[str] = None, # not used yet
        text_masks = None,
        # 1.1 pick one of the following, text_con_input>text_embeds
        text_con_input = None, # main channel for text
        text_embeds = None,    # backdoor channel: for test only
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
        cond_scale = 7.5,   
        # Researcher Netruk44 have reported 5-10 to be optimal, but anything greater than 10 to break.
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
        '''
        Goal: produce sample output based on
        1. text_embeds
        2. cond_img
        '''
        # ++
        if self.CKeys['Debug_Level']==PD_Forw_Level:
            print (f"In ProteinDesigner.sample() ...")
        
        device = default(device, self.device)
        
        # 1. on text_emb channel:
        # ................................................................
        # input: AA seq: (batch, text_len)
        # target: text_embeds
        # shape: (batch, text_len, text_emb_dim)
        assert not ((text_con_input!=None) & (text_embeds!=None)), \
        "text_embeds and text_con_input cannot be provided at the same time"
        
        if (text_con_input!=None):
            
            # 1.1. convert AA_seq into hidden_state
            text_con_input, seq_logits =\
            self.map_seq_tok_to_logits_and_hidden_state_w_pLM(
                text_con_input,
            ) 
            # hidden state: (batch, dim_hidden, seq_len)
            # normalized logits: (batch, dim_logits, seq_len)
            
            text_con_input = rearrange(text_con_input, 'b m n -> b n m')
            # (batch, seq_len, dim_hidden)
            
            # 1.2. expand to text_con_embeds
            # For Predictor, self.to_text_con_emb() is Identity()
            text_con_embeds = self.to_text_con_emb(text_con_input)
            # (batch, seq_len, text_con_embeds_dim)
            
            # ++
            if self.CKeys['Debug_Level']==PD_Samp_Level:
                print (f"text_con_embeds.shape: {text_con_embeds.shape}")
                
        # 1.3. add int text_con_pos part
        if text_con_embeds != None:
            
            batch_size = text_con_embeds.shape[0]
            
        # if text_con_input != None:
            # 1.3.1 check the length of the text_embeds
            assert text_con_embeds.shape[1]<=self.text_max_len, "text_embed length is larger than defined text_max_len"
            # check the shape of the input text_con_embeds: (batch, text_len, embeds)
            if text_con_embeds.shape[1]<self.text_max_len:
                # need to change the length: assume text_emb padding is zero
                text_con_embeds_fixed = torch.zeros(
                    text_con_embeds.shape[0],
                    self.text_max_len,
                    text_con_embeds.shape[2],
                ).to(self.diffuser_core.device)
                text_con_embeds_fixed[:,:text_con_embeds.shape[1],:]=text_con_embeds
                text_con_embeds = text_con_embeds_fixed
            #
            # 1.3.2 provide positional embedding: text_pos_matrix_i
            if self.text_pos_matrix_i is None:
                # for Protein Predictor
                text_pos_embeds = torch.zeros(
                    batch_size,
                    self.text_max_len,
                    0,
                ).to(self.diffuser_core.device)
                # (b, text_len, 0), only a trivial one
                pass
            else:
                # kept for Protein Designer
                text_pos_mat_i_ = self.text_pos_matrix_i.repeat(
                    text_con_embeds.shape[0], 1
                ).to(device) # (batch, text_len)
                text_pos_embeds = self.to_text_pos_emb(text_pos_mat_i_) 
                # (batch, text_len, text_pos_emb_dim)
            # 
            # 1.3 merge con and pos
            text_embeds = torch.cat( (text_con_embeds, text_pos_embeds),2 )
            # ++
            if self.CKeys['Debug_Level']==PD_Samp_Level:
                print (f"text_pos_embeds.shape: {text_pos_embeds.shape}")
                print (f"text_con_embeds.shape: {text_con_embeds.shape}")
                print (f"After merge, text_embeds.shape: {text_embeds.shape}")
                
        # 2. on cond_img channel: AA seq (b, seq_len)
        # ................................................................
        if cond_images.dim() == 2:
            cond_images_embeds = seq_logits
            # (b, dim_logits, seq_len)
            # ++
            if self.CKeys['Debug_Level']==PD_Samp_Level:
                if cond_images!=None:
                    print (f"cond_images_embeds.shape: {cond_images_embeds.shape}")
        
        
        # pass into the diffuser:EImagen
        output = self.diffuser_core.sample(
            # 1. on text condition
            texts  = texts,
            text_masks = text_masks,
            text_embeds = text_embeds,
            # 2. on condition images
            cond_images = cond_images_embeds, # cond_images,# for PPredictor
            cond_video_frames = cond_video_frames,
            post_cond_video_frames = post_cond_video_frames,
            # 3. inpaint images
            inpaint_videos = inpaint_videos,
            inpaint_images = inpaint_images,
            inpaint_masks = inpaint_masks,
            inpaint_resample_times = inpaint_resample_times,
            # 
            init_images = init_images,
            skip_steps = skip_steps,
            sigma_min = sigma_min,
            sigma_max = sigma_max,
            video_frames = video_frames,
            batch_size = batch_size,
            cond_scale = cond_scale,
            lowres_sample_noise_level = lowres_sample_noise_level,
            start_at_unet_number = start_at_unet_number,
            start_image_or_video = start_image_or_video,
            stop_at_unet_number = stop_at_unet_number,
            return_all_unet_outputs = return_all_unet_outputs,
            return_pil_images = return_pil_images,
            use_tqdm = use_tqdm,
            use_one_unet_in_gpu = use_one_unet_in_gpu,
            device = device,
        )
        # output: (batch, )
        
        
        return output
    
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
    @torch.no_grad()
    @eval_decorator
    def sample_to_NMS_list (
        self,
        # add some
        mask_from_Y=None,
        NormFac_list=None,
        # for self.sample method
        # ===============================
        # 1. on text condition
        texts: List[str] = None, # not used yet
        text_masks = None,
        # 1.1 pick one of the following, text_con_input>text_embeds
        text_con_input = None, # main channel for text
        text_embeds = None,    # backdoor channel: for test only
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
        cond_scale = 7.5,   
        # Researcher Netruk44 have reported 5-10 to be optimal, but anything greater than 10 to break.
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
        '''
        Goal: produce sample output based on
        1. text_embeds
        2. cond_img
        '''
        
        # 1. call the diffusion core
        output_diffuser = self.sample(
            # 1. on text condition
            texts,
            text_masks = text_masks,
            # 1.1 pick one of the following, text_con_input>text_embeds
            text_con_input = text_con_input, # main channel for text
            text_embeds = text_embeds,    # backdoor channel: for test only
            # 2. on condition images
            cond_images = cond_images,
            cond_video_frames = cond_video_frames,
            post_cond_video_frames = post_cond_video_frames,
            # 3. inpaint images
            inpaint_videos = inpaint_videos,
            inpaint_images = inpaint_images,
            inpaint_masks = inpaint_masks,
            inpaint_resample_times = inpaint_resample_times,
            # 
            init_images = init_images,
            skip_steps = skip_steps,
            sigma_min = sigma_min,
            sigma_max = sigma_max,
            video_frames = video_frames,
            batch_size = batch_size,
            cond_scale = cond_scale,   
            # Researcher Netruk44 have reported 5-10 to be optimal, but anything greater than 10 to break.
            lowres_sample_noise_level = lowres_sample_noise_level,
            start_at_unet_number = start_at_unet_number,
            start_image_or_video = start_image_or_video,
            stop_at_unet_number = stop_at_unet_number,
            return_all_unet_outputs = return_all_unet_outputs,
            return_pil_images = return_pil_images,
            use_tqdm = use_tqdm,
            use_one_unet_in_gpu = use_one_unet_in_gpu,
            device = device,
        )
        
        # get dim
        n_samp = output_diffuser.shape[0]
        n_mode = output_diffuser.shape[1]
        seq_len = output_diffuser.shape[2]
        
        # 2. translate NMS vectors in a batch back into a list
        if mask_from_Y is None:
            print (f"No mask is provided as input.")
            # assume all is to keep
            result_mask = torch.ones(
                (n_samp, seq_len), 
                dtype=torch.bool
            ) # all true
        else:
            result_mask = mask_from_Y
        
        # 3. translate result back to a list
        # print (n_samp)
        
        nms_vecs_list = get_nms_vec_as_arr_list_from_batch_using_mask(
            result_mask,     # (b, seq_len) # torch.tensor
            output_diffuser, # (b, n_mode, seq_len)
            NormFac_list,    # (n_mode, )
        )
        
        # # ----------------------------------
        # nms_vecs_list = []
        # for i_samp in range(n_samp):
        #     this_mask = result_mask[i_samp] # (seq_len, )
        #     this_nms_vecs = output_diffuser[i_samp] # (n_mode, seq_len)
        #     # 
        #     this_nms_arr = []
        #     for i_mode in range(n_mode):
        #         this_add = this_nms_vecs[i_mode][this_mask==True] # only work for 1D tensor
        #         this_add = this_add * NormFac_list[i_mode]
        #         # print (this_add.shape)
        #         this_nms_arr.append(
        #             this_add.cpu().detach().numpy()
        #         )
        #     this_nms_arr = np.array(this_nms_arr)
        #     #
        #     nms_vecs_list.append(this_nms_arr)
                
        
        return nms_vecs_list
    
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    
    # get shape-mask based on toks in Y
    def read_mask_from_seq_toks_using_pLM(
        self,
        #
        batch_Y, # (batch, seq_len)
    ):
        # in ESM, looks like 0 xxx 2, 1,1,...1
       
        
        mask_from_Y = torch.logical_and(
            batch_Y!=esm_tok_to_idx['<cls>'],
            batch_Y!=esm_tok_to_idx['<eos>']
        )
        mask_from_Y = torch.logical_and(
            mask_from_Y,
            batch_Y!=esm_tok_to_idx['<pad>']
        )
        
        return mask_from_Y
    
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

    # translate Cond in the batch (b, seq_len) back to 
    def map_batch_idx_into_AA_idx_and_toks_string_lists_w_pLM(
        self,
        # 
        Y_train_batch, # (b, seq_len)
        mask_from_Y=None,   # (b, seq_len)
    ):
        n_case = Y_train_batch.shape[0]
        
        AA_idx_arr_list = []
        AA_string_list = []
        
        for i_case in range(n_case):
            this_idx = Y_train_batch[i_case] # (seq_len, )
            
            if not (mask_from_Y is None):
                this_mask = mask_from_Y[i_case]
                this_idx = this_idx[this_mask==True] # work only for 1D tensor
            
            this_idx_arr = this_idx.cpu().detach().numpy()
            # 
            AA_idx_arr_list.append(this_idx_arr)
            
            # translate to seq
            this_seq = []
            for ii in range(len(this_idx_arr)):
                this_seq.append(
                    self.pLM_alphabet.get_tok(
                        this_idx_arr[ii]
                    )
                )
            this_seq_string = "".join(this_seq)
            #
            AA_string_list.append(this_seq_string)
        
        return AA_idx_arr_list, AA_string_list
            