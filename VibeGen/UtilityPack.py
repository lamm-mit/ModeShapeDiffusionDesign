# ==========================================================
# Utility functions
# ==========================================================
import os
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
import numpy as np
import math
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBList

import torch
from einops import rearrange
import esm

import json

# =========================================================
# 
def Print(this_line):
    # may update for multi-core case later
    print (this_line)

def print_dict_content(this_dict):
    
    for this_key in this_dict.keys():
        print (f"    {this_key}: {this_dict[this_key]}")
        
# =========================================================
# create a folder path if not exist
def create_path(this_path):
    if not os.path.exists(this_path):
        print('Creating the given path...')
        os.mkdir (this_path)
        path_stat = 1
        print('Done.')
    else:
        print('The given path already exists!')
        path_stat = 2
    return path_stat

# ============================================================
# on esm, rebuild AA sequence from embedding
# ============================================================

def decode_one_ems_token_rec(this_token, esm_alphabet):
    # print( (this_token==esm_alphabet.cls_idx).nonzero(as_tuple=True)[0] )
    # print( (this_token==esm_alphabet.eos_idx).nonzero(as_tuple=True)[0] )
    # print( (this_token==100).nonzero(as_tuple=True)[0]==None )

    id_b=(this_token==esm_alphabet.cls_idx).nonzero(as_tuple=True)[0]
    id_e=(this_token==esm_alphabet.eos_idx).nonzero(as_tuple=True)[0]
    
    
    if len(id_e)==0:
        # no ending for this one, so id_e points to the end
        id_e=len(this_token)
    else:
        id_e=id_e[0]
    if len(id_b)==0:
        id_b=0
    else:
        id_b=id_b[-1]

    this_seq = []
    # this_token_used = []
    for ii in range(id_b+1,id_e,1):
        # this_token_used.append(this_token[ii])
        this_seq.append(
            esm_alphabet.get_tok(this_token[ii])
        )
        
    this_seq = "".join(this_seq)

    # print(this_seq)    
    # print(len(this_seq))
    # # print(this_token[id_b+1:id_e]) 
    return this_seq


def decode_many_ems_token_rec(batch_tokens, esm_alphabet):
    rev_y_seq = []
    for jj in range(len(batch_tokens)):
        # do for one seq: this_seq
        this_seq = decode_one_ems_token_rec(
            batch_tokens[jj], esm_alphabet
            )
        rev_y_seq.append(this_seq)
    return rev_y_seq

def Print_model_params (model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    Print (
        f"Total model parameters: {pytorch_total_params}\nTrainable parameters: {pytorch_total_params_trainable}\n"
    )

def get_model_params (model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    resu = {
        'tot': pytorch_total_params,
        'trainable': pytorch_total_params_trainable,
        'freezed': pytorch_total_params-pytorch_total_params_trainable
    }
    
    return resu

def write_one_line_to_file(
    this_line,
    file_name,
    mode,
    accelerator=None
):
    with open(file_name, mode) as f:
        f.write(this_line)
        
# ==============================================================
# 
# def convert_into_tokens_using_prob(
#     prob_result, 
#     pLM_Model_Name
# ):
#     if pLM_Model_Name=='esm2_t33_650M_UR50D' \
#     or pLM_Model_Name=='esm2_t36_3B_UR50D'   \
#     or pLM_Model_Name=='esm2_t30_150M_UR50D' \
#     or pLM_Model_Name=='esm2_t12_35M_UR50D' :
        
#         repre=rearrange(
#             prob_result,
#             'b c l -> b l c'
#         )
#         # with torch.no_grad():
#         #     logits=model.lm_head(repre) # (b, l, token_dim)
#         logits = repre
            
#         tokens=logits.max(2).indices # (b,l)
        
#     else:
#         print("pLM_Model is not defined...")
#     return tokens,logits

def read_mask_from_input(
    # consider different type of inputs
    # raw data: x_data (sequences)
    # tokenized: x_data_tokenized
    tokenized_data=None, # X_train_batch, 
    mask_value=None,
    seq_data=None,       # Y_train_batch,
    max_seq_length=None,
):
    # # old:
    # mask = X_train_batch!=mask_value
    # new
    if seq_data!=None:
        # use the real sequence length to create mask
        n_seq = len(seq_data)
        mask = torch.zeros(n_seq, max_seq_length)
        for ii in range(n_seq):
            this_len = len(seq_data[ii])
            mask[ii,1:1+this_len]=1
        mask = mask==1
    #
    elif tokenized_data!=None:
        n_seq = len(tokenized_data)
        mask = tokenized_data!=mask_value
        # fix the beginning part: 0+content+00, not 00+content+00
        for ii in range(n_seq):
            # get all nonzero index
            id_1 = (mask[ii]==True).nonzero(as_tuple=True)[0]
            # correction for ForcPath, 
            # pick up 0.0 for zero-force padding at the beginning
            mask[ii,1:id_1[0]]=True
    
    return mask

# on pLM tokens
# basic 20 in abr order: ARNDCEQGHILKMFPSTWYV
# in esm, tot = 33
# basic 20 in esm order: LAGVSERTIDPKQNFYMHWC
# others (4): <cls> <pad> <eos> <unk>
# special (9): X B U Z O . - <null_1> <mask>
# LAGVSERTIDPKQNFYMHWC: toke the channels: 4-23
# full dict
esm_tok_to_idx = \
{'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}

esm_idx_to_tok = \
{'0': '<cls>', '1': '<pad>', '2': '<eos>', '3': '<unk>', '4': 'L', '5': 'A', '6': 'G', '7': 'V', '8': 'S', '9': 'E', '10': 'R', '11': 'T', '12': 'I', '13': 'D', '14': 'P', '15': 'K', '16': 'Q', '17': 'N', '18': 'F', '19': 'Y', '20': 'M', '21': 'H', '22': 'W', '23': 'C', '24': 'X', '25': 'B', '26': 'U', '27': 'Z', '28': 'O', '29': '.', '30': '-', '31': '<null_1>', '32': '<mask>'}

common_AA_list = "LAGVSERTIDPKQNFYMHWC"


# common_AA_idx_in_esm = []
# for ii in range(len(common_AA_list)):
#     common_AA_idx_in_esm.append(
#         esm_tok_to_idx[
#             common_AA_list[ii]
#         ]
#     )

common_AA_idx_in_esm = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

def keep_only_20AA_channels_in_one_pLM_logits(
    full_logits, # (seq_len, channel)
    keep_channels=common_AA_idx_in_esm
):
    assert full_logits.shape[-1]==33, \
    "Not ESM logits shape"
    
    n_channel = full_logits.shape[-1]
    for this_c in range(n_channel):
        if not (this_c in keep_channels):
            full_logits[:,this_c]=-float('inf')
            
    return full_logits

def get_toks_list_from_Y_batch(
    batch_GT,     # (b, seq_len)
    batch_mask,   # (b, seq_len)
):
    toks_list = []
    seqs_list = []
    
    for ii in range(len(batch_GT)):
        this_GT = batch_GT[ii]
        this_mask = batch_mask[ii]
        this_GT = this_GT[this_mask==True]
        # 
        toks_list.append(this_GT)
        this_seq = [
            esm_idx_to_tok[str(jj.item())] for jj in this_GT
        ]
        this_seq = "".join(this_seq)
        seqs_list.append(this_seq)
        
        
    return toks_list, seqs_list

def compare_two_seq_strings(seq_PR, seq_GT):
    # take seq_GT as the ref, 
    # assume len(seq_GT)>=len(seq_PR)
    len_comp = min( len(seq_PR), len(seq_GT))
    num_hit = 0
    for ii in range(len_comp):
        if seq_PR[ii]==seq_GT[ii]:
            num_hit += 1
    ratio_hit = num_hit/len_comp
    
    return ratio_hit

def save_2d_tensor_as_np_arr_txt(
    X_tensor, # (a, b)
    mask = None, # (b)
    outname = None,
):
    assert X_tensor.dim() == 2
    
    if not (mask is None):
        assert mask.dim() == 1
    
    if not (mask is None):
        X_tensor = X_tensor[:, mask]

        
    test_one_X_arr = X_tensor.cpu().detach().numpy()
    if outname is None:
        print (test_one_X_arr)
    else:
        np.savetxt(outname, test_one_X_arr)
    # # to read back as a 2d np arr
    # test_one_X_arr_1 = np.loadtxt(test_file)
    
# ++ read back for checking
def read_2d_np_arr_from_txt(
    test_file
):
    test_one_X_arr_1 = np.loadtxt(test_file)
    return test_one_X_arr_1 
    
def string_diff (seq1, seq2):    
    return   sum(1 for a, b in zip(seq1, seq2) if a != b) + abs(len(seq1) - len(seq2))

# def write_fasta_file(
#     this_seq, 
#     this_head, 
#     this_file
# ):
#     with open(this_file, mode = 'w') as f:
#         f.write (f">{this_head}\n")
#         f.write (f"{this_seq}")
        
def write_fasta_file(
    this_seq_list, 
    this_head_list, 
    this_file
):
    n_seq = len(this_seq_list)
    
    with open(this_file, mode = 'w') as f:
        for i_seq in range(n_seq):
            
            f.write (f">{this_head_list[i_seq]}\n")
            f.write (f"{this_seq_list[i_seq]}\n")

# ++
def read_recover_AAs_only(test_fasta_file):

    file1 = open(test_fasta_file, 'r')
    Lines = file1.readlines()
    # only get AA
    AA_GT = Lines[1].strip()
    AA_recon_GT = Lines[3].strip()
    
    resu = {}
    resu['AA_GT'] = AA_GT
    resu['AA_recon_GT'] = AA_recon_GT
    
    return resu

# ===================================================================================
# old one
def fold_one_AA_to_SS_using_omegafold_for_5_Diffusionfold(
    sequence,
    num_cycle=16,
    device=None,
    # ++++++++++++++
    prefix=None,
    AA_file_path=None,  
    PDB_file_path=None, # output file path
    head_note=None,
):
    AA_file_name = f"{AA_file_path}/{prefix}_.fasta"
    print ("Writing FASTA file: ", AA_file_name)
    head_line = f"{head_note}"
    with open (AA_file_name, mode ='w') as f:
        f.write (f'>{head_line}\n')
        f.write (f'{sequence}')
    # 
    # 
    PDB_result=f"{PDB_file_path}/{head_line}.pdb"
    if not os.path.exists(PDB_result):
        print (f"Now run OmegaFold.... on device={device}")    
        # !omegafold $filename $prefix --num_cycle $num_cycle --device=$device
        cmd_line=F"omegafold {AA_file_name} {PDB_file_path} --num_cycle {num_cycle} --device={device}"
        print(os.popen(cmd_line).read())

        print ("Done OmegaFold")

        # PDB_result=f"{prefix}{OUTFILE}.PDB"
        
        print (f"Resulting PDB file...:  {PDB_result}")
    else:
        print (f"PDB file already exist.")
    
    return PDB_result, AA_file_name
# 
# ===================================================================================
# new one: need to install the modified omegafold from self-hold repo
# https://github.com/Bo-Ni/OmegaFold_0.git
def get_subbatch_size(L):
    if L <  500: return 500
    if L < 1000: return 500 # 500 # 200
    return 150

def fold_one_AA_to_SS_using_omegafold(
    sequence,
    num_cycle=16,
    device=None,
    # ++++++++++++++
    prefix="Temp", # None,
    AA_file_path="./",  # None,  
    PDB_file_path="./", # output file path
    head_note="Temp_", # None,
):
    AA_file_name = f"{AA_file_path}/{prefix}_.fasta"
    print ("Writing FASTA file: ", AA_file_name)
    head_line = f"{head_note}"
    with open (AA_file_name, mode ='w') as f:
        f.write (f'>{head_line}\n')
        f.write (f'{sequence}')
    # 
    subbatch_size = get_subbatch_size(len(sequence))
    # 
    PDB_result=f"{PDB_file_path}/{head_line}.pdb"
    
    if not os.path.exists(PDB_result):
        Print (f"Now run OmegaFold.... on device={device}\n\n")    
        # !omegafold $filename $prefix --num_cycle $num_cycle --device=$device
        # cmd_line=F"omegafold {AA_file_name} {PDB_file_path} --num_cycle {num_cycle} --device={device}"
        cmd_line=F"omegafold {AA_file_name} {PDB_file_path} --subbatch_size {str(subbatch_size)} --num_cycle {num_cycle} --device={device}"
        
        Print(os.popen(cmd_line).read())

        Print ("Done OmegaFold")

        # PDB_result=f"{prefix}{OUTFILE}.PDB"
        
        Print (f"Resulting PDB file...:  {PDB_result}\n\n")
    else:
        Print (f"PDB file already exist.")
    
    return PDB_result, AA_file_name
# 
# ===================================================================================
# plot
import py3Dmol

def plot_plddt_legend(dpi=100):
    thresh = ['plDDT:','Very low (<50)','Low (60)','OK (70)','Confident (80)','Very high (>90)']
    plt.figure(figsize=(1,0.1),dpi=dpi)
    ########################################
    for c in ["#FFFFFF","#FF0000","#FFFF00","#00FF00","#00FFFF","#0000FF"]:
        plt.bar(0, 0, color=c)
    plt.legend(thresh, frameon=False,
             loc='center', ncol=6,
             handletextpad=1,
             columnspacing=1,
             markerscale=0.5,)
    plt.axis(False)
    return plt

color = "lDDT" # choose from ["chain", "lDDT", "rainbow"]
show_sidechains = False #choose from {type:"boolean"}
show_mainchains = False #choose from {type:"boolean"}

def show_pdb(
    pdb_file, 
    flag=0,   
    show_sidechains=False, 
    show_mainchains=False, 
    color="lDDT"
):
    model_name = f"Flag_{flag}"
    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js',)
    view.addModel(open(pdb_file,'r').read(),'pdb')

    if color == "lDDT":
        view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':50,'max':90}}})
    elif color == "rainbow":
        view.setStyle({'cartoon': {'color':'spectrum'}})
    elif color == "chain":
        chains = len(queries[0][1]) + 1 if is_complex else 1
        for n,chain,color in zip(
            range(chains),list("ABCDEFGH"),
            ["lime","cyan","magenta","yellow","salmon","white","blue","orange"]
        ):
            view.setStyle({'chain':chain},{'cartoon': {'color':color}})
    
    if show_sidechains:
        BB = ['C','O','N']
        view.addStyle({'and':[{'resn':["GLY","PRO"],'invert':True},{'atom':BB,'invert':True}]},
                    {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
        view.addStyle({'and':[{'resn':"GLY"},{'atom':'CA'}]},
                    {'sphere':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
        view.addStyle({'and':[{'resn':"PRO"},{'atom':['C','O'],'invert':True}]},
                        {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})  
    if show_mainchains:
        BB = ['C','O','N','CA']
        view.addStyle({'atom':BB},{'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})

    view.zoomTo()
    if color == "lDDT":
        plot_plddt_legend().show() 
    # 
    return view
# 
# ===================================================================================
# SecStr
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBList

Unique_SecStr_Q8_String="HET~BGIS"
Unique_SecStr_Q3_String="HEC"
# 
# =============================================
# count statistics of Q8 based SecStr
# 
def count_ratio_for_Q8(
    this_secstr,
    Unique_SecStr_Q8_String=Unique_SecStr_Q8_String,
):
    resu = {}
    seq_len = len(this_secstr)
    for this_char in Unique_SecStr_Q8_String:
        resu[this_char] = this_secstr.count(this_char)/seq_len
    #
    return resu
# =============================================
# count statistics of Q3 based SecStr
# 
def count_ratio_for_Q3(
    this_secstr,
    Unique_SecStr_Q3_String=Unique_SecStr_Q3_String,
):
    resu = {}
    seq_len = len(this_secstr)
    for this_char in Unique_SecStr_Q3_String:
        resu[this_char] = this_secstr.count(this_char)/seq_len
    #
    return resu
# ===============================================
# 
def analyze_SS_Q8_Q3_for_df(
    df_smo_recon_BSDB_4P_expanded,
    Unique_SecStr_Q8_String=Unique_SecStr_Q8_String,
    Unique_SecStr_Q3_String=Unique_SecStr_Q3_String,
):
    # 
    # do statistics on Q8
    this_key_to_add = 'stat_Q8'
    if not (this_key_to_add in df_smo_recon_BSDB_4P_expanded.keys()):
        print (f"Add new key {this_key_to_add}")
        df_smo_recon_BSDB_4P_expanded[this_key_to_add] = df_smo_recon_BSDB_4P_expanded.apply(
            # ================ change this part ===========================
            lambda row: count_ratio_for_Q8(
                row['SS_Q8'],
                Unique_SecStr_Q8_String=Unique_SecStr_Q8_String,
            ),
            # ================ change this part ===========================
            axis=1,
        )

    # do statistics on Q3
    this_key_to_add = 'stat_Q3'
    if not (this_key_to_add in df_smo_recon_BSDB_4P_expanded.keys()):
        print (f"Add new key {this_key_to_add}")
        df_smo_recon_BSDB_4P_expanded[this_key_to_add] = df_smo_recon_BSDB_4P_expanded.apply(
            # ================ change this part ===========================
            lambda row: count_ratio_for_Q3(
                row['SS_Q3'],
                Unique_SecStr_Q3_String=Unique_SecStr_Q3_String,
            ),
            # ================ change this part ===========================
            axis=1,
        )
    # 
    # expand to df columns
    for this_char in Unique_SecStr_Q3_String:
        print (f"working on Q3 {this_char}")
        this_key_to_add = 'stat_Q3_'+this_char
        if not (this_key_to_add in df_smo_recon_BSDB_4P_expanded.keys()):
            print (f"Add new key {this_key_to_add}")
            df_smo_recon_BSDB_4P_expanded[this_key_to_add] = df_smo_recon_BSDB_4P_expanded.apply(
                # ================ change this part ===========================
                lambda row: row['stat_Q3'][this_char],
                # ================ change this part ===========================
                axis=1,
            )
    # expand to Q8
    for this_char in Unique_SecStr_Q8_String:
        print (f"working on Q8 {this_char}")
        this_key_to_add = 'stat_Q8_'+this_char
        if not (this_key_to_add in df_smo_recon_BSDB_4P_expanded.keys()):
            print (f"Add new key {this_key_to_add}")
            df_smo_recon_BSDB_4P_expanded[this_key_to_add] = df_smo_recon_BSDB_4P_expanded.apply(
                # ================ change this part ===========================
                lambda row: row['stat_Q8'][this_char],
                # ================ change this part ===========================
                axis=1,
            )
            
    return df_smo_recon_BSDB_4P_expanded
# ==================================================

def get_DSSP_result (fname):
    pdb_list = [fname]

    # parse structure
    p = PDBParser()
    for i in pdb_list:
        structure = p.get_structure(i, fname)
        # use only the first model
        model = structure[0]
        # calculate DSSP
        dssp = DSSP(model, fname, file_type='PDB' )
        # extract sequence and secondary structure from the DSSP tuple
        sequence = ''
        sec_structure = ''
        for z in range(len(dssp)):
            a_key = list(dssp.keys())[z]
            sequence += dssp[a_key][1]
            sec_structure += dssp[a_key][2]

        # print extracted sequence and structure
        #print(i)
        #print(sequence)
        #print(sec_structure)
        #
        # The DSSP codes for secondary structure used here are:
        # =====     ====
        # Code      Structure
        # =====     ====
        # H         Alpha helix (4-12)
        # B         Isolated beta-bridge residue
        # E         Strand
        # G         3-10 helix
        # I         Pi helix
        # T         Turn
        # S         Bend
        # -         None
        # =====     ====
        #

        sec_structure = sec_structure.replace('-', '~')
        sec_structure_3state=sec_structure


        # if desired, convert DSSP's 8-state assignments into 3-state [C - coil, E - extended (beta-strand), H - helix]
        sec_structure_3state = sec_structure_3state.replace('~', 'C')
        sec_structure_3state = sec_structure_3state.replace('I', 'C')
        sec_structure_3state = sec_structure_3state.replace('T', 'C')
        sec_structure_3state = sec_structure_3state.replace('S', 'C')
        sec_structure_3state = sec_structure_3state.replace('G', 'H')
        sec_structure_3state = sec_structure_3state.replace('B', 'E')
        
    return sec_structure,sec_structure_3state, sequence

# ++ for postprocess
def get_DSSP_set_result(fname):
    sec_structure,sec_structure_3state, sequence = get_DSSP_result (fname)
    resu={}
    resu['SecStr_Q8']=sec_structure
    resu['SecStr_Q3']=sec_structure_3state
    resu['AA_from_DSSP']=sequence
    
    return resu

def write_DSSP_result_to_json(
    sec_structure,
    sec_structure_3state,
    sequence,
    filename,
):
    resu = {
        "Q8": sec_structure,
        "Q3": sec_structure_3state,
        "AA_from_DSSP": sequence
    }
    resu_json = json.dumps(resu, indent=4)
    
    with open(filename, "w") as f:
        f.write(resu_json)
        
#     # to read back
#     with open(filename, 'r') as openfile:
#         # Reading from json file
#         json_object = json.load(openfile)
        
#     print(json_object)
#     print(type(json_object)) # dict

# ==============================================================
# pick some Normal Mode from a df
# For NMS vectors only
def build_XCond_list_from_df(
    df,
    key_list,
    pick_id_list,
):
    n_mode = len(key_list)
    n_samp = len(pick_id_list)
    resu = []
    for id_samp in pick_id_list:
        this_X_list = []
        for this_key in key_list:
            add_one = df[this_key].values[id_samp]
            
            this_X_list.append(
                add_one
            )
        this_X = np.array(this_X_list)
        resu.append(this_X)
        
    return resu

# For AA Seq only
def build_AA_list_from_df(
    df,
    AA_key,
    pick_id_list,
):
    n_samp = len(pick_id_list)
    resu = []
    for id_samp in pick_id_list:
        resu.append(
            df[AA_key].values[id_samp]
        )
        
    return resu

# ==============================================================
# add for Protein Predictor
def get_nms_vec_as_arr_list_from_batch_using_mask(
    result_mask,     # (b, seq_len) # torch.tensor
    output_diffuser, # (b, n_mode, seq_len)
    NormFac_list,    # (n_mode, )
):
    n_samp = output_diffuser.shape[0]
    n_mode = output_diffuser.shape[1]
    
    nms_vecs_list = []
    for i_samp in range(n_samp):
        this_mask = result_mask[i_samp] # (seq_len, )
        this_nms_vecs = output_diffuser[i_samp]
        
        # to take care of multi-modes
        this_nms_arr = []
        for i_mode in range(n_mode):
            this_add = this_nms_vecs[i_mode][this_mask==True] # only work for 1D tensor
            this_add = this_add * NormFac_list[i_mode] # map it back to real values
            this_nms_arr.append(
                this_add.cpu().detach().numpy()
            )
        this_nms_arr = np.array(this_nms_arr) # convert into np.arr
        
        # deliver to the list to store
        nms_vecs_list.append(this_nms_arr)
        
    return nms_vecs_list

# compare two nms_vecs
def compare_two_nms_vecs_arr(
    PR_nms_vecs, 
    GT_nms_vecs,
):
    n_mode = GT_nms_vecs.shape[0]
    # calculate error for each mode and the tot
    # calculate rela_L2 error
    resu = {}
    for i_mode in range(n_mode):
        resu["rela_L2_err_Mode_"+str(i_mode)]=np.linalg.norm(PR_nms_vecs[i_mode]-GT_nms_vecs[i_mode])/np.linalg.norm(GT_nms_vecs[i_mode])
    # 
    # calculate for multi-modes
    resu["rela_L2_err_MulMode"]=np.linalg.norm(PR_nms_vecs-GT_nms_vecs)/np.linalg.norm(GT_nms_vecs)
    
    return resu

# ======================================================

def translate_seqs_list_into_idx_tensor_w_pLM(
    # 1. model converter
    esm_batch_converter,
    AA_seq_max_len,
    # 2. on input
    raw_condition_list,
    device
):
    
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
    
    y_data = y_data.to(device)
    
    return y_data

# ==================================================

# def cal_err_list_using_