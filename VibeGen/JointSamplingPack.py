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
import pandas as pd

from ema_pytorch import EMA

from einops import rearrange

from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem

import shutil
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

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
    create_path,
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
    compare_two_nms_vecs_arr,
    translate_seqs_list_into_idx_tensor_w_pLM
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
def merge_two_topk(
    y_goo,
    y_bad,
):
    y={}
    y['indices']=torch.concatenate(
        (y_goo.indices,y_bad.indices)
    )
    y['values']= torch.concatenate(
        (y_goo.values,y_bad.values)
    )
    
    len_goo = len(y_goo.indices)
    len_bad = len(y_bad.indices)
    name_list_goo = [f'min_err_{ii}' for ii in range(len_goo)]
    name_list_bad = [f'max_err_{ii}' for ii in range(len_bad)]
    name_list = name_list_goo+name_list_bad
    
    y['name_type']=name_list
    
    # indices=torch.concatenate(
    #     (y1.indices,y2.indices)
    # )
    # values= torch.concatenate(
    #     (y1.values,y2.values)
    # )
    # y=torch.return_types.topk_out(
    #     values=values, 
    #     indices=indices
    # )
    return y

# //////////////////////////////////////////////////////////////
# 5. Main class/functions: a base trainer wrap for ProteinDesigner
# //////////////////////////////////////////////////////////////
        
        
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# joint PD & PP for sampling

def joint_sampling_w_PD_and_PP(
    # 1. model
    PD_wk_ProteinDesigner,
    PP_wk_ProteinPredictor,
    # 2. data
    PD_test_set_condition_list, # input as a list of NMS vecs
    PD_test_set_AA_list=None,   # whether GT is provided
    PD_DataKeys=None,
    # 3. control param
    n_try_w_PD = 100,  # For PD, try this number times as a batch
    n_keep_w_PP_goo = 2,   # Use PP to pick the top this_number samples
    # ++
    n_keep_w_PP_bad=2, # also keep the worest one as ref:
    #
    PD_cond_scal = 7.5,
    PP_cond_scal = 7.5,
    # 4. outputs
    joint_sampling_dir = None,
    joint_sampling_prefix = f"TestSet_", # 3. on postprocessing
    # 
    IF_plot_PP = True,
    IF_showfig = True,
    IF_save_pred_pack = True,
    IF_plot_PD = True,
    IF_fold_seq = True,
    IF_show_foldding = True,
    IF_DSSP = True,
    # others
    device = None,
):
    
    if not (PD_test_set_AA_list is None):
        assert len(PD_test_set_condition_list)==len(PD_test_set_AA_list), \
        "the input Conditioning and GT don't have the same length..."
    else:
        Print(f"Only input Conditioning is provided...")
        
    # prepare wk dir
    if not os.path.exists(joint_sampling_dir):
        Print(f"Create joint sampling path...")
        create_path(joint_sampling_dir)
        Print(f"Done.")
    else:
        Print(f"Dir exists. Use caution...")
    
    # model status
    PD_wk_ProteinDesigner.turn_on_eval_mode()
    PP_wk_ProteinPredictor.turn_on_eval_mode()
    
    
    # prepare
    # on PD
    text_len_max = PD_wk_ProteinDesigner.text_max_len
    img_len_max = PD_wk_ProteinDesigner.diffuser_core.image_sizes[0]
    len_in = min(text_len_max, img_len_max) # depend on problem statement, may change
    
    text_embed_input_dim = PD_wk_ProteinDesigner.text_embed_input_dim
    cond_img_channels = PD_wk_ProteinDesigner.diffuser_core.unets[0].cond_images_channels
    mode_in = min(text_embed_input_dim, cond_img_channels)
    
    print (len_in)
    print (mode_in)
    
    # on PP
    AA_seq_max_len = PP_wk_ProteinPredictor.seq_obj_max_size
    esm_batch_converter = PP_wk_ProteinPredictor.pLM_alphabet.get_batch_converter(
        truncation_seq_length=AA_seq_max_len-2
    )
    PP_wk_ProteinPredictor.pLM.eval()
    
    PR_err_for_PP = torch.nn.MSELoss(reduction='none')
    
    n_keep_w_PP = n_keep_w_PP_goo+n_keep_w_PP_bad
    
    # ++ 
    # ++ get GT for PD if exists
    # 6. translate back to a batch for PP
    if not (PD_test_set_AA_list is None):
        GT_test_set_AA_batch_for_PD = \
        translate_seqs_list_into_idx_tensor_w_pLM(
            # 1. model converter
            esm_batch_converter,
            AA_seq_max_len,
            # 2. on input
            raw_condition_list=PD_test_set_AA_list,
            # 3. on outpt
            device=device
        ) # (batch, seq_len)
        # 
        # get len info as shape mask
        mask_from_Y_all = PD_wk_ProteinDesigner.read_mask_from_seq_toks_using_pLM(
            GT_test_set_AA_batch_for_PD
        )
        # 
        GT_idx_list, GT_seqs_list = get_toks_list_from_Y_batch(
            GT_test_set_AA_batch_for_PD,
            mask_from_Y_all
        )
    else:
        GT_test_set_AA_batch_for_PD = None
        mask_from_Y_all = None
        GT_idx_list = None
        GT_seqs_list = None
        
    # ++ for picking up from the previous runs
    # .............................................................
    reco_csv = joint_sampling_dir+'/'+joint_sampling_prefix+\
    f'Try_{n_try_w_PD}_Pick_{n_keep_w_PP}'+'_reco.csv'
    
    Print (f"Use reco file: \n{reco_csv}")
    if not os.path.isfile(reco_csv):
        # first time
        Print (f"First run of the sampling...\n\n")
        # write the top line
        csv_top_line = f"root_path,error_L2,r2"
        write_one_line_to_file(
            this_line=csv_top_line+'\n',
            file_name=reco_csv,
            mode='w',
        )
        n_pick_finished = 0
        n_samp_finished = 0
    else:
        df_reco = pd.read_csv(reco_csv)
        n_pick_finished = len(df_reco)//n_keep_w_PP
        n_samp_finished = len(df_reco)%n_keep_w_PP
        Print (f"Previously, finished input #: {n_pick_finished}")
        Print (f"finished samp #: {n_samp_finished}\n\n")
    
        
    
    X_file_list = []
    
    # pick one sample
    for i_pick in range(len(PD_test_set_condition_list)):
        
        if i_pick > n_pick_finished-1: # pick up from the previous
            
            Print (f"\n\nWorking on Input #: {i_pick}\n\n")
            
            # i_pick = 1
            
            # 1. get X data padded
            X_arr = np.zeros(
                (mode_in, len_in)
            ) # (n_mode, seq_len)

            for j in range(mode_in):
                X_arr[j, :] = pad_a_np_arr_esm_for_NMS(
                    PD_test_set_condition_list[i_pick][j, :],
                    0,
                    len_in
                )
            print (X_arr.shape)

            # 2. get X normalized and formated
            for j in range(mode_in):
                X_arr[j, :] = X_arr[j, :]/PD_DataKeys['Xnormfac'][j]

            X_train = torch.from_numpy(X_arr).float() # (c, seq_len)

            # 3. expand into a batch
            X_train = X_train.unsqueeze(0).repeat(n_try_w_PD,1,1)

            print (X_train.shape)
            
            X_train_batch = X_train.to(device)
            
            # 4. prep the GT for NMS vecs
            seq_len_pick = PD_test_set_condition_list[i_pick].shape[1]
            GT_NMS_tensor_pick = torch.from_numpy(
                PD_test_set_condition_list[i_pick]
            ).float() # (n_mode, this_seq_len)
            GT_NMS_tensor = GT_NMS_tensor_pick.unsqueeze(0).repeat(
                n_try_w_PD,1,1
            ) # (batch, n_mode, this_seq_len)
            GT_NMS_tensor = GT_NMS_tensor.to(device)

            # 5. make prediction w. PD
            print (f"\n\nPD making {str(n_try_w_PD)} designs ...\n\n")
            
            PR_toks_list, PR_seqs_list, result_mask = \
            PD_wk_ProteinDesigner.sample_to_pLM_idx_seq(
                # 
                common_AA_only=True, # False,
                mask_from_Y=None, # mask_from_Y
                # if none, will use mask from X, cond_img then text
                # 
                text_con_input = X_train_batch,
                cond_images = X_train_batch,
                # 
                cond_scale = PD_cond_scal,
            )
            # result_mask: (batch, seq_len)
            
            # 6. translate back to a batch for PP
            print (f"\n\nPP predicting performances ...\n\n")
            
            y_data_for_PP = \
            translate_seqs_list_into_idx_tensor_w_pLM(
                # 1. model converter
                esm_batch_converter,
                AA_seq_max_len,
                # 2. on input
                raw_condition_list=PR_seqs_list,
                # 3. on outpt
                device=device
            ) # (batch, seq_len)
            
            print (y_data_for_PP.shape)
            
            # 7. make prediction w PP
            PR_NMS_arr_list = PP_wk_ProteinPredictor.sample_to_NMS_list(
                # mask
                mask_from_Y = result_mask,
                NormFac_list = PD_DataKeys['Xnormfac'],
                # 
                text_con_input = y_data_for_PP,
                cond_images = y_data_for_PP,
                # 
                cond_scale = PP_cond_scal,
            ) # list (n_mode, seq_len)
            # make the list into a tensor
            PR_NMS_tensor = torch.from_numpy(
                np.stack(PR_NMS_arr_list, axis=0) # (b, n_mode, this_seq_len)
            ).float()
            PR_NMS_tensor = PR_NMS_tensor.to(device)
                
            # 8. calc the error for NMS vecs
            
            PR_NMS_err_batch = PR_err_for_PP(
                PR_NMS_tensor,
                GT_NMS_tensor, 
            ) # (b, n_mode, this_seq_len)
            PR_NMS_err_batch = torch.sum(
                PR_NMS_err_batch,
                dim=(1,2)
            ) # (b, )
            
            print (f"Pick the best {n_keep_w_PP_goo}...")
            
            idxs_vals_to_pick_goo = torch.topk(
                PR_NMS_err_batch,
                k=n_keep_w_PP_goo,
                largest=False,
            )
            # have indices and values
            # (n_keep_w_PP, )
            print (f"Pick the worst {n_keep_w_PP_bad}...")
            idxs_vals_to_pick_bad = torch.topk(
                PR_NMS_err_batch,
                k=n_keep_w_PP_bad,
                largest=True,
            )
            
            print (f"N\n\now, fold the picked best {n_keep_w_PP_goo} and worst {n_keep_w_PP_bad} samples...")
            idxs_vals_to_pick = merge_two_topk(
                y_goo=idxs_vals_to_pick_goo,
                y_bad=idxs_vals_to_pick_bad,
            )
            
            
            # 9. postprocess
            for i_in_k in range(n_keep_w_PP):
                
                if i_in_k > n_samp_finished-1: # pick up from the previous
                    
                    Print (f"\n\nProcessing Picked #: Input {i_pick+1} -- Design {i_in_k+1}\n\n")
                    if i_in_k<n_keep_w_PP_goo:
                        IF_goo = 1
                        Print (f"This supposed to be the {i_in_k+1}th among the best")
                    else:
                        Print (f"This supposed to be the {i_in_k-n_keep_w_PP_goo+1}th among the worst")
                    Print (f"\n\n")
                        
                
                    # i_case_in_batch = idxs_vals_to_pick.indices[i_in_k].item()
                    # ++
                    i_case_in_batch = idxs_vals_to_pick['indices'][i_in_k].item()
                    i_case_type_name = idxs_vals_to_pick['name_type'][i_in_k]

                    print (f"NMS Err based on PP: {idxs_vals_to_pick['values'][i_in_k]}")
                    print (f"Idx in PD batch: {i_case_in_batch}")

                    outname_prefix=joint_sampling_dir+'/'+joint_sampling_prefix \
                    +'InputX_'+str(i_pick) \
                    +'_TopSamp_'+str(i_in_k)+'_'+i_case_type_name+'_'

                    # 9.1 plot the GT and PP_PR
                    if IF_plot_PP:
                        Print(f"Plot the PP comparison...")

                        fig=plt.figure()
                        for i_mode in range(mode_in):
                            # PR
                            plt.plot (
                                PR_NMS_tensor[i_case_in_batch][i_mode].cpu().detach().numpy(),
                                '^-',
                                label= f'Predicted mode {i_mode+1}',
                            )
                            # GT
                            plt.plot (
                                GT_NMS_tensor[i_case_in_batch][i_mode].cpu().detach().numpy(),
                                '--',
                                label= f'GT mode {i_mode+1}',
                            )

                        plt.legend()
                        plt.xlabel(f"AA #")
                        plt.ylabel(f"Vibrational Disp. amp")
                        outname = outname_prefix+"_VibDisAmp_PP.jpg"
                        if IF_showfig==1:
                            plt.show()
                        else:
                            plt.savefig(outname, dpi=200)
                        plt.close ()
                        outname = None

                    # 9.2 things to print: seq and recovery ratio
                    # 
                    Print(
                        f"PR Seq:\n{PR_seqs_list[i_case_in_batch]}"
                    )
                    if not (GT_seqs_list is None):
                        Print(
                            f"GT Seq:\n{GT_seqs_list[i_pick]}"
                        )
                        # 
                        hit_ratio_PR_GT = compare_two_seq_strings(
                            PR_seqs_list[i_case_in_batch], # Prediction
                            GT_seqs_list[i_pick], # GT
                        )
                        Print(
                            f"Recovery ratio: {hit_ratio_PR_GT}"
                        )
                    # 9.3 save basics: input X, GT Y and PR Y
                    # 9.3.1 input X
                    if IF_save_pred_pack:
                        
                        # ...........................................
                        # keep a record of X Input
                        Print (f"\nSave conditioning...\n")
                        
                        X_train_print = X_train_batch[i_case_in_batch] # (n_mode, seq_len)
                        X_train_print = X_train_print[:, result_mask[i_case_in_batch]]
                        
                        Print (f"{X_train_print.shape}")
                        
                        outname = outname_prefix+"_XCond.txt"
                        # save_2d_tensor_as_np_arr_txt(
                        #     X_train_batch[i_case_in_batch],
                        #     mask = result_mask[i_case_in_batch],
                        #     # set to None to turn off
                        #     outname=outname,
                        # )
                        save_2d_tensor_as_np_arr_txt(
                            X_train_print,
                            mask = None,
                            # set to None to turn off
                            outname=outname,
                        )
                        X_file_list.append(outname)
                        outname = None
                        
                        # ...........................................
                        # keep a record of PP prediction
                        Print (f"\nSave PP pred conditioning...\n")
                        Print (f"{PR_NMS_tensor[i_case_in_batch].shape}")
                        
                        outname = outname_prefix+"_PR_XCond.txt"
                        save_2d_tensor_as_np_arr_txt(
                            PR_NMS_tensor[i_case_in_batch], # from (b, n_mode, this_seq_len)
                            mask = None, # set to None to turn off
                            outname=outname,
                        )
                        X_file_list.append(outname)
                        outname = None
                        
                        # ..........................................
                        # compare X_Input and X_PR_Input
                        this_r2_list = []
                        for i_mode in range(mode_in):
                            this_r2 = r2_score(
                                # 
                                y_true=PR_NMS_tensor[i_case_in_batch][i_mode]
                                .cpu().detach().numpy(),
                                
                                y_pred=X_train_print[i_mode]
                                .cpu().detach().numpy(),
                            )
                            
                            this_r2_list.append(
                                this_r2
                            )
                            # Print (f"Mode {i_mode+1} r2: {this_r2}\n\n")
                            
                        this_r2_mean = np.mean(this_r2_list)
                        # Print (f"Ave r2 over modes: {this_r2_mean}")
                        # ++ Pearson coef
                        this_rou_list = []
                        for i_mode in range(mode_in):
                            this_rou = pearsonr(
                                # 
                                PR_NMS_tensor[i_case_in_batch][i_mode]
                                .cpu().detach().numpy(),
                                
                                X_train_print[i_mode]
                                .cpu().detach().numpy(),
                            )[0]
                            
                            this_rou_list.append(
                                this_rou
                            )
                            Print (f"Mode {i_mode+1} \u03C1: {this_rou}\n\n")
                            
                        this_rou_mean = np.mean(this_rou_list)
                        Print (f"Ave \u03C1 over modes: {this_rou_mean}")
                        
                    else:
                        # 
                        this_r2_mean = -100
                        

                    # 9.3.2 on GT Y: skipped
                    if not (GT_seqs_list is None):
                        
                        Print (f"\nSave GT AA\n")
                        Print (f"seq len: {len(GT_seqs_list[i_pick])}")
                        
                        # make record of GT seq
                        this_label = f"GT"
                        outname = outname_prefix+"_GT.fasta"
                        write_fasta_file(
                            this_seq_list=[GT_seqs_list[i_pick]], 
                            this_head_list=[this_label], 
                            this_file=outname
                        )
                        this_label = None
                        outname = None


                    # 9.3.3 on PR Y: sequence
                    
                    Print (f"\nSave PR AA\n")
                    Print (f"seq len: {len(PR_seqs_list[i_case_in_batch])}")
                    
                    this_label = f"Recovery_ratio_PR_GT: {hit_ratio_PR_GT}"
                    outname = outname_prefix+"_PR.fasta"
                    write_fasta_file(
                        this_seq_list=[PR_seqs_list[i_case_in_batch]], 
                        this_head_list=[this_label], 
                        this_file=outname
                    )
                    this_label = None
                    outname = None


                    # 9.3 plot the PD_PR
                    if IF_plot_PD:
                        
                        Print(f"\nPlot the PD comparison...\n")

                        fig=plt.figure()
                        plt.plot (
                            PR_toks_list[i_case_in_batch].cpu().detach().numpy(),
                            '^-',
                            label= f'Predicted',
                        )
                        # ++
                        if not (GT_idx_list is None):
                            plt.plot (
                                GT_idx_list[i_pick].cpu().detach().numpy(),
                                '^-',
                                label= f'GT',
                            )
                        plt.legend()
                        plt.xlabel(f"AA #")
                        plt.ylabel(f"idx in ESM")
                        outname = outname_prefix+"_AA_idx_comp_PD.jpg"
                        if IF_showfig==1:
                            plt.show()
                        else:
                            plt.savefig(outname, dpi=200)
                        plt.close ()
                        outname = None

                    if IF_fold_seq:
                        # 
                        Print (f"\nFolding the predicted AA...\n")
                        #
                        outname = outname_prefix+"_PR.pdb"
                        # 3.1 fold AA using Omegaofld
                        # ........................................
                        PR_PDB_temp, PR_AA_temp = \
                        fold_one_AA_to_SS_using_omegafold(
                            sequence=PR_seqs_list[i_case_in_batch],
                            num_cycle=16,
                            device=device,
                            # ++++++++++++++
                            prefix="Temp", # None,
                            AA_file_path=joint_sampling_dir,
                            # sample_dir, # "./",  # None,  
                            PDB_file_path=joint_sampling_dir,
                            # sample_dir, # "./", # output file path
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
                        if IF_show_foldding:

                            Print (f"\n\nShow the folded structure...")
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
                            if IF_DSSP:

                                Print (f"\n\nAna. secondary structure...")
                                # apply dssp
                                PR_DSSP_Q8, PR_DSSP_Q3, PR_AA_from_DSSP = \
                                get_DSSP_result(
                                    outname
                                )
                                # check if any AA is missing
                                if len(PR_AA_from_DSSP)!=len(PR_seqs_list[i_case_in_batch]):
                                    Print (f"Missing AA during DSSP. Use caution.")
                                # print result
                                block_to_print = f"""\
AA:           {PR_seqs_list[i_case_in_batch]}
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


                # for i_in_k in range(n_keep_w_PP):
                    # ++ add a finish record
                    # csv_top_line = f"root_path,error_L2"
                    this_w_line = f"{outname_prefix},{idxs_vals_to_pick['values'][i_in_k].item()},{this_r2_mean}"
                    write_one_line_to_file(
                        this_line=this_w_line+'\n',
                        file_name=reco_csv,
                        mode='a',
                    )
                    
                # ..................................................
                # once this is called, reset n_samp_finished
                # ONLY run once
                n_samp_finished = 0
                # if i_in_k > n_samp_finished-1: # pick up from the previous
                
        else:
            pass # this record is already finished

    
