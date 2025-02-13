from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import Tokenizer

from torch.utils.data import DataLoader,Dataset
import pandas as pd
import seaborn as sns

import torchvision
 
import matplotlib.pyplot as plt
import numpy as np
 
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from functools import partial, wraps

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler

from matplotlib.ticker import MaxNLocator

import torch

import esm

# special packages

import VibeGen.UtilityPack as UPack
from VibeGen.UtilityPack import (
    decode_one_ems_token_rec,
    decode_many_ems_token_rec
)

# 
DPack_Random = 123456

class RegressionDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)

# ============================================================
# handle NMA result
# 
# 1. screen the dataset
# ============================================================
def screen_dataset_MD_NMS_MultiModes(
    # # --
    # file_path,
    # ++
    csv_file=None,
    pk_file =None,
    PKeys=None,
    CKeys=None,
):
    # unload the parameters
    
    store_path = PKeys['data_dir']
    IF_SaveFig = CKeys['SilentRun']
    min_AASeq_len = PKeys['min_AA_seq_len']
    max_AASeq_len = PKeys['max_AA_seq_len']
    max_used_Seg_Num = PKeys['max_used_Seg_Num']
    
    # max_used_Smo_F = PKeys['max_Force_cap']
    
    # working part
    if csv_file != None:
        # not used for now
        # functions
        print('=============================================')
        print('1. read in the csv file...')
        print('=============================================')
        arr_key = PKeys['arr_key']
        
        df_raw = pd.read_csv(csv_file)
        UPack.Print("Raw df has keys:")
        UPack.Print(df_raw.keys())

        # convert string array back to array
        for this_key in arr_key:
            # np.array(list(map(float, one_record.split(" "))))
            df_raw[this_key] = df_raw[this_key].apply(lambda x: np.array(list(map(float, x.split(" ")))))
        # =====================================================
        # adjust if needed
        # patch up
        df_raw.rename(columns={"sample_FORCEpN_data":"sample_FORCE_data"}, inplace=True)
        print('Updated keys: \n', df_raw.keys())
    
    elif pk_file != None:
        # functions
        print('=============================================')
        print('1. read in the pk file...')
        print('=============================================')
        # 
        df_raw = pd.read_pickle(pk_file)
        
        UPack.Print("Raw df has keys:")
        UPack.Print(df_raw.keys())
    
    # ..............................................................................
    # --
    fig = plt.figure(figsize=(24,16),dpi=200)
    fig, ax0 = plt.subplots()
    for ii in range(len( df_raw )):
        if df_raw['AA_Eff_Len'][ii]<=6400:
    #         # +
    #         ax0.plot(
    #             df_disp_forc_smo['normalized_pull_gap_data'][ii], 
    #             df_disp_forc_smo['forc_data'][ii],
    #             color="blue",label='full data'
    #         )
    #         #
            ax0.plot(
                df_raw['Norm_Resi_Ind_List'][ii], 
                # df_raw['sample_FORCEpN_data'][ii], 
                df_raw['Mode7_NormDisAmp'][ii], 
                alpha=0.1,
                # color="green",label='simplified data',
                # linestyle='None',marker='^'
            )
            # ============================================
            # # too slow to do this
            # ax0.scatter(
            #     df_raw['NormResiIndx_At_MaxVibrAmp_Mode7'][ii], 
            #     df_raw['NormDisAmp_At_MaxVibrAmp_Mode7'][ii], 
            # )
        else:
            print(df_raw['pdb_id'][ii])
            # we see mistakes in: 1. wrong len of the AA; 2. wrong # of residue of the beginning and end
    plt.xlabel('Normalized residue index')
    plt.ylabel('Normalized vibrational disp. amp.')
    outname = store_path+'CSV_0_NMS_Mode7_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    plt.close()
    
    print('=============================================')
    print('2. screen the entries...')
    print('=============================================')
    #
    df_isnull = pd.DataFrame(
        round(
            (df_raw.isnull().sum().sort_values(ascending=False)/df_raw.shape[0])*100,
            1
        )
    ).reset_index()
    df_isnull.style.format({'% of Missing Data': lambda x:'{:.1%}'.format(abs(x))})
    cm = sns.light_palette("skyblue", as_cmap=True)
    df_isnull = df_isnull.style.background_gradient(cmap=cm)
    print('Check null...')
    print( df_isnull )
    
    print('Working on a dataframe with useful keywords')
    # suppose to be a smaller one
    # Focus on mode 7 For the moment
    # Expand to modes 7,8,9 
    protein_df = pd.DataFrame().assign(
        pdb_id=df_raw['pdb_id'],
        AA=df_raw['AA_Full'], 
        seq_len=df_raw['AA_Eff_Len'],
        AA_Seg_Num=df_raw['AA_Seg_Num'],
        Norm_Resi_Ind_List=df_raw['Norm_Resi_Ind_List'],
        # on mode 7
        Mode7_NormDisAmp=df_raw['Mode7_NormDisAmp'],
        ScaFac_7=df_raw['ScaFac_7'],
        Mode7_NormDis=df_raw['Mode7_NormDis'],
        Mode7_Freq=df_raw['Mode7_Freq'],
        NormResiIndx_At_MaxVibrAmp_Mode7=df_raw['NormResiIndx_At_MaxVibrAmp_Mode7'],
        NormDisAmp_At_MaxVibrAmp_Mode7=df_raw['NormDisAmp_At_MaxVibrAmp_Mode7'],
        Mode7_FixLen_NormDisAmp=df_raw['Mode7_FixLen_NormDisAmp'],
        # on mode 8
        Mode8_NormDisAmp=df_raw['Mode8_NormDisAmp'],
        ScaFac_8=df_raw['ScaFac_8'],
        Mode8_NormDis=df_raw['Mode8_NormDis'],
        Mode8_Freq=df_raw['Mode8_Freq'],
        NormResiIndx_At_MaxVibrAmp_Mode8=df_raw['NormResiIndx_At_MaxVibrAmp_Mode8'],
        NormDisAmp_At_MaxVibrAmp_Mode8=df_raw['NormDisAmp_At_MaxVibrAmp_Mode8'],
        # Mode8_FixLen_NormDisAmp=df_raw['Mode8_FixLen_NormDisAmp'],
        # on mode 9
        Mode9_NormDisAmp=df_raw['Mode9_NormDisAmp'],
        ScaFac_9=df_raw['ScaFac_9'],
        Mode9_NormDis=df_raw['Mode9_NormDis'],
        Mode9_Freq=df_raw['Mode9_Freq'],
        NormResiIndx_At_MaxVibrAmp_Mode9=df_raw['NormResiIndx_At_MaxVibrAmp_Mode9'],
        NormDisAmp_At_MaxVibrAmp_Mode9=df_raw['NormDisAmp_At_MaxVibrAmp_Mode9'],
        # Mode9_FixLen_NormDisAmp=df_raw['Mode9_FixLen_NormDisAmp'],
    )
    # ++ add new keys on energy if needed
    
    # screen using AA length
    print('a. screen using sequence length...')
    print('original sequences #: ', len(protein_df))
    #
    protein_df.drop(
        protein_df[protein_df['seq_len']>max_AASeq_len-2].index, 
        inplace = True
    )
    protein_df.drop(
        protein_df[protein_df['seq_len'] <min_AASeq_len].index, 
        inplace = True
    )
    protein_df=protein_df.reset_index(drop=True)
    print('used sequences #: ', len(protein_df))
    
    print('b. screen using Seg_Num values...')
    print('original sequences #: ', len(protein_df))
    #
    protein_df.drop(
        protein_df[protein_df['AA_Seg_Num']>max_used_Seg_Num].index, 
        inplace = True
    )
    # protein_df.drop(
    #     protein_df[protein_df['seq_len'] <min_AA_len].index, 
    #     inplace = True
    # )
    protein_df=protein_df.reset_index(drop=True)
    print('afterwards, sequences #: ', len(protein_df))
    
    UPack.Print("Some statisitcs of the selected records")
    # ===================================================
    # some figures to compare the screening effect
    # ===================================================
    # fig = plt.figure(figsize=(12,8),dpi=200)
    fig = plt.figure()
    # sns.displot(
    #     data= protein_df,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.displot(
    #     data= df_raw,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.distplot(
    #     protein_df['seq_len'],
    #     binw
    #     bins=50,kde=False, 
    #     rug=False,norm_hist=False)
    # sns.distplot(
    #     df_raw['AA_Eff_Len'],
    #     bins=50,kde=False, 
    #     rug=False,norm_hist=False)
    
    sns.histplot(
        df_raw,
        x="AA_Eff_Len",
        binwidth=1,
        # bins=50,
        # kde=False, 
        # rug=False,
        # norm_hist=False,
    )
    sns.histplot(
        protein_df,
        x="seq_len",
        binwidth = 1,
        # bins=50,
        # kde=False, 
        # rug=False,
        #norm_hist=False,
    )
    #
    plt.legend(['Full recrod','Selected'])
    plt.xlabel('AA length')
    outname = store_path+'CSV_1_AALen_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
    
    fig = plt.figure()
    # sns.displot(
    #     data= protein_df,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.displot(
    #     data= df_raw,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.distplot(df_raw['Max_Smo_Force'],bins=50,kde=False, rug=False,norm_hist=False)
    # sns.distplot(protein_df['Max_Smo_Force'],bins=50,kde=False, rug=False,norm_hist=False)
    # # --
    # sns.distplot(protein_df['NormDisAmp_At_MaxVibrAmp_Mode7'],kde=True, rug=False,norm_hist=False)
    # sns.distplot(df_raw['NormDisAmp_At_MaxVibrAmp_Mode7'],kde=True, rug=False,norm_hist=False)
    # ++
    sns.histplot(
        df_raw['NormDisAmp_At_MaxVibrAmp_Mode7'],
        kde=True, 
        # rug=False,
        # norm_hist=False
    )
    sns.histplot(
        protein_df['NormDisAmp_At_MaxVibrAmp_Mode7'],
        kde=True, 
        # rug=False,
        # norm_hist=False
    )
    #
    plt.legend(['Full recrod','Selected',])
    plt.xlabel('Normalized Max. Disp. Amp.')
    outname = store_path+'CSV_2_MaxNormVibrAmp_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
        #
    # fig = plt.figure(figsize=(12,8),dpi=200)
    fig = plt.figure()
    # sns.displot(
    #     data= protein_df,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.displot(
    #     data= df_raw,
    #     x="seq_len", kde=False,bins=50,
    # )
    # #
    # sns.distplot(df_raw['Max_Smo_Force'],bins=50,kde=False, rug=False,norm_hist=False)
    # sns.distplot(protein_df['Max_Smo_Force'],bins=50,kde=False, rug=False,norm_hist=False)
    # # --
    # sns.distplot(protein_df['NormResiIndx_At_MaxVibrAmp_Mode7'],kde=True, rug=False,norm_hist=False)
    # sns.distplot(df_raw['NormResiIndx_At_MaxVibrAmp_Mode7'],kde=True, rug=False,norm_hist=False)
    # ++
    sns.histplot(
        df_raw['NormResiIndx_At_MaxVibrAmp_Mode7'],
        kde=True, 
        # rug=False,norm_hist=False
    )
    sns.histplot(
        protein_df['NormResiIndx_At_MaxVibrAmp_Mode7'],
        kde=True, 
        # rug=False,norm_hist=False
    )
    #
    plt.legend(['Full recrod','Selected',])
    plt.xlabel('Normalized residue index at the Max. virbational Disp. Amp.')
    outname = store_path+'CSV_3_NormResiInd_at_MaxNormVibrAmp_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
    
    print('=============================================')
    print('3. summary statistics...')
    print('=============================================')
    # re-plot the simplified results of SMD
    print('Check selected in NMS records...')

    fig = plt.figure(figsize=(24,16),dpi=200)
    fig, ax0 = plt.subplots()
    for ii in range(len( protein_df )):
        if protein_df['seq_len'][ii]<=6400:
    #         # +
    #         ax0.plot(
    #             df_disp_forc_smo['normalized_pull_gap_data'][ii], 
    #             df_disp_forc_smo['forc_data'][ii],
    #             color="blue",label='full data'
    #         )
    #         #
            ax0.plot(
                protein_df['Norm_Resi_Ind_List'][ii], 
                # protein_df['sample_FORCEpN_data'][ii], 
                protein_df['Mode7_NormDisAmp'][ii], 
                alpha=0.1,
                # color="green",label='simplified data',
                # linestyle='None',marker='^'
            )
            #
            # too slow
            # 
            # ax0.scatter(
            #     protein_df['NormResiIndx_At_MaxVibrAmp_Mode7'][ii], 
            #     protein_df['NormDisAmp_At_MaxVibrAmp_Mode7'][ii], 
            # )
            
        else:
            print(protein_df['pdb_id'][ii])
            # we see mistakes in: 1. wrong len of the AA; 2. wrong # of residue of the beginning and end
    plt.xlabel('Normalized residue index')
    plt.ylabel('Normalized vibrational displacement amplitude')
    plt.title('Mode 7')
    outname = store_path+'CSV_4_Screened_NMS_Mode7_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    plt.close()
    # ==========================================================
    fig = plt.figure(figsize=(24,16),dpi=200)
    fig, ax0 = plt.subplots()
    for ii in range(len( protein_df )):
        if protein_df['seq_len'][ii]<=6400:
    #         # +
    #         ax0.plot(
    #             df_disp_forc_smo['normalized_pull_gap_data'][ii], 
    #             df_disp_forc_smo['forc_data'][ii],
    #             color="blue",label='full data'
    #         )
    #         #
            ax0.plot(
                protein_df['Norm_Resi_Ind_List'][ii], 
                # protein_df['sample_FORCEpN_data'][ii], 
                protein_df['Mode8_NormDisAmp'][ii], 
                alpha=0.1,
                # color="green",label='simplified data',
                # linestyle='None',marker='^'
            )
            #
            # too slow
            # 
            # ax0.scatter(
            #     protein_df['NormResiIndx_At_MaxVibrAmp_Mode7'][ii], 
            #     protein_df['NormDisAmp_At_MaxVibrAmp_Mode7'][ii], 
            # )
            
        else:
            print(protein_df['pdb_id'][ii])
            # we see mistakes in: 1. wrong len of the AA; 2. wrong # of residue of the beginning and end
    plt.xlabel('Normalized residue index')
    plt.ylabel('Normalized vibrational displacement amplitude')
    plt.title('Mode 8')
    outname = store_path+'CSV_4_Screened_NMS_Mode8_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    plt.close()
    # ==========================================================
    fig = plt.figure(figsize=(24,16),dpi=200)
    fig, ax0 = plt.subplots()
    for ii in range(len( protein_df )):
        if protein_df['seq_len'][ii]<=6400:
    #         # +
    #         ax0.plot(
    #             df_disp_forc_smo['normalized_pull_gap_data'][ii], 
    #             df_disp_forc_smo['forc_data'][ii],
    #             color="blue",label='full data'
    #         )
    #         #
            ax0.plot(
                protein_df['Norm_Resi_Ind_List'][ii], 
                # protein_df['sample_FORCEpN_data'][ii], 
                protein_df['Mode9_NormDisAmp'][ii], 
                alpha=0.1,
                # color="green",label='simplified data',
                # linestyle='None',marker='^'
            )
            #
            # too slow
            # 
            # ax0.scatter(
            #     protein_df['NormResiIndx_At_MaxVibrAmp_Mode7'][ii], 
            #     protein_df['NormDisAmp_At_MaxVibrAmp_Mode7'][ii], 
            # )
            
        else:
            print(protein_df['pdb_id'][ii])
            # we see mistakes in: 1. wrong len of the AA; 2. wrong # of residue of the beginning and end
    plt.xlabel('Normalized residue index')
    plt.ylabel('Normalized vibrational displacement amplitude')
    plt.title('Mode 9')
    outname = store_path+'CSV_4_Screened_NMS_Mode9_Dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    plt.close()
    # ==========================================================
    
    
    fig = plt.figure()
    # ++
    # sns.histplot(
    #     df_raw['NormDisAmp_At_MaxVibrAmp_Mode7'],
    #     kde=True, 
    #     # rug=False,
    #     # norm_hist=False
    # )
    sns.histplot(
        protein_df['NormDisAmp_At_MaxVibrAmp_Mode7'],
        kde=True, 
        # rug=False,
        # norm_hist=False
        label='Mode 7'
    )
    sns.histplot(
        protein_df['NormDisAmp_At_MaxVibrAmp_Mode8'],
        kde=True, 
        # rug=False,
        # norm_hist=False
        label='Mode 8',
        alpha=0.5,
    )
    sns.histplot(
        protein_df['NormDisAmp_At_MaxVibrAmp_Mode9'],
        kde=True, 
        # rug=False,
        # norm_hist=False
        label='Mode 9',
        alpha=0.1,
    )
    #
    # plt.legend(['Full recrod','Selected',])
    plt.legend()
    plt.xlabel('Normalized Max. Disp. Amp.')
    outname = store_path+'CSV_5_Screen_MaxNormVibrAmp_Dist_Mode7_8_9.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    
    
    
    # fig = plt.figure(figsize=(12,8),dpi=200)
    fig = plt.figure()
    # ++
    sns.histplot(
        protein_df['NormResiIndx_At_MaxVibrAmp_Mode7'],
        kde=True, 
        # rug=False,norm_hist=False
        label='Mode 7',
    )
    # sns.distplot(
    #     df_raw['NormResiIndx_At_MaxVibrAmp_Mode7'],
    #     kde=True, 
    #     # rug=False,norm_hist=False
    # )
    sns.histplot(
        protein_df['NormResiIndx_At_MaxVibrAmp_Mode8'],
        kde=True, 
        # rug=False,norm_hist=False
        label='Mode 8',
        alpha=0.5,
    )
    # 
    sns.histplot(
        protein_df['NormResiIndx_At_MaxVibrAmp_Mode9'],
        kde=True, 
        # rug=False,norm_hist=False
        label='Mode 9',
        alpha=0.1,
    )
    #
    plt.legend()
    # plt.legend(['Full recrod','Selected',])
    plt.xlabel('Normalized residue index at the Max. virbational Disp. Amp.')
    outname = store_path+'CSV_6_Screen_NormResiInd_at_MaxNormVibrAmp_Dist_Mode7_8_9.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:  
        plt.show()
    #
    plt.close()
    # =========================================================
    
    
    
    print('Done')
    
    
    return df_raw, protein_df

# =============================================================
# ++ for NMS data: N in length, not N+1
def pad_a_np_arr_esm_for_NMS(x0,add_x,n_len):
    # in the screening part, we have made len(x0)+2<=n_len
    # for NMS, need to add a add_x to the 0th position for <col>
    # 
    x1 = x0.copy()
    x1 = np.insert(x1,0,add_x)
    n0 = len(x1)
    if n0<n_len:
        for ii in range(n0,n_len,1):
            # x1.append(add_x)
            x1= np.append(x1, [add_x])
    else:
        print('No padding is needed')

    
    
    # # print(n0)
    # x1 = x0.copy()
    # # print('copy x0: ', x1)
    # # print('x0 len: ', len(x1))
    # # # add one for <cls>
    # # x1 = [add_x]+x1 # somehow, this one doesn't work
    # # print(x1)
    # # print('x1 len: ',len(x1) )
    # n0 = len(x1)
    # #
    # if n0<n_len:
    #     for ii in range(n0,n_len,1):
    #         # x1.append(add_x)
    #         x1= np.append(x1, [add_x])
    # else:
    #     print('No padding is needed')
        
    return x1

# =============================================================
# calc normalization factors for X
def prepare_X_and_Y_from_df_NMS_pLM_MultiMode(
    protein_df,
    PKeys,
    CKeys,
):
    # unload the parameters
    store_path = PKeys['data_dir']
    IF_SaveFig = CKeys['SilentRun']
    
    X_Keys = PKeys['X_Key'] # now X_Key is a list
    max_AA_len = PKeys['max_AA_seq_len']
    
    # working
    UPack.Print("======================================================")
    UPack.Print("1. work on X data: Normalized NormVibrAmp")
    UPack.Print("======================================================")
    
    n_X_Key = len(X_Keys)
    n_record = len(protein_df)
    UPack.Print(f"Find X_Keys #: {n_X_Key}")
    UPack.Print(f"Find protein record #: {n_record}")
    #
    X = np.zeros(
        (n_record, n_X_Key, max_AA_len)
    ) # n_rec, n_mode, seq_len
    # in the last dim, need to pad the record as: 0, real_seq, 0,..,0
    for i in range(n_record):
        for ii, this_X_Key in enumerate(X_Keys):
            X[i,ii,:] = pad_a_np_arr_esm_for_NMS(
                protein_df[this_X_Key][i],
                0,
                max_AA_len
            )
    UPack.Print (f"X.shape: {X.shape}")
    
    # do a quick check
    i_X_Example = 10
    print (f"To check, pick one example {protein_df['pdb_id'][i_X_Example]}")
    print (f"Find sequence length: {protein_df['seq_len'][i_X_Example]}")
    print (f"Correspondingly, X data:")
    for ii, this_X_Key in enumerate(X_Keys):
        this_print_arr = X[i_X_Example,ii,:]
        print (f"For {this_X_Key},\n{this_print_arr}")
        # check zero padding
        jj_0 = 0
        jj_1 = 0
        for jj in range(max_AA_len):
            if np.fabs(this_print_arr[jj])>0:
                jj_0 = jj
                break
        for jj in range(max_AA_len):
            if np.fabs(this_print_arr[-(jj+1)])>0:
                jj_1 = jj
                break
        jj_data_len = max_AA_len-jj_0-jj_1
        print (f" Begin_padding: {jj_0}")
        print (f" End_padding: {jj_1}")
        print (f" Data_len: {jj_data_len}")
    
    UPack.Print (f"Now, calculate Normalization Factor for each mode")
    UPack.Print (f"Upper bound of the NFs: {np.amax(X)}")
    
    X_NF_List = []
    for ii, this_X_Key in enumerate(X_Keys):
        this_x_max = np.amax(
            X[:,ii,:]
        )
        X_NF_List.append(this_x_max)
        # normalization
        X[:,ii,:] = X[:,ii,:]/this_x_max
        
    UPack.Print (f"X_NF_List: {X_NF_List}")
    
    UPack.Print("======================================================")
    UPack.Print("2. work on Y data: AA Sequence")
    UPack.Print("======================================================")
    # take care of the y part: AA encoding
    #create and fit tokenizer for AA sequences
    seqs = protein_df.AA.values
    # ++ for pLM: esm
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print("pLM model: ", PKeys['ESM-2_Model'])
    
    if PKeys['ESM-2_Model']=='esm2_t33_650M_UR50D':
        # print('Debug block')
        # embed dim: 1280
        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
    elif PKeys['ESM-2_Model']=='esm2_t12_35M_UR50D':
        # embed dim: 480
        esm_model, esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
    elif PKeys['ESM-2_Model']=='esm2_t36_3B_UR50D':
        # embed dim: 2560
        esm_model, esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        len_toks=len(esm_alphabet.all_toks)
    elif PKeys['ESM-2_Model']=='esm2_t30_150M_UR50D':
        # embed dim: 640
        esm_model, esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        len_toks=len(esm_alphabet.all_toks)
    else:
        print("protein language model is not defined.")
    #
    # for check
    print("esm_alphabet.use_msa: ", esm_alphabet.use_msa)
    print("# of tokens in AA alphabet: ", len_toks)
    # need to save 2 positions for <cls> and <eos>
    esm_batch_converter = esm_alphabet.get_batch_converter(
        truncation_seq_length=PKeys['max_AA_seq_len']-2
    )
    esm_model.eval()  # disables dropout for deterministic results
    # prepare seqs for the "esm_batch_converter..."
    # add dummy labels
    seqs_ext=[]
    for i in range(len(seqs)):
        seqs_ext.append(
            (" ", seqs[i])
        )
    # batch_labels, batch_strs, batch_tokens = esm_batch_converter(seqs_ext)
    _, y_strs, y_data = esm_batch_converter(seqs_ext)
    y_strs_lens = (y_data != esm_alphabet.padding_idx).sum(1)
    # print(batch_tokens.shape)
    print ("y_data.dim: ", y_data.dtype)   
        
    fig_handle = sns.histplot(
        data=pd.DataFrame({'AA code': np.array(y_data).flatten()}),
        x='AA code', 
        bins=np.array([i-0.5 for i in range(0,33+3,1)]), # np.array([i-0.5 for i in range(0,20+3,1)])
        # binwidth=1,
    )
    fig = fig_handle.get_figure()
    fig_handle.set_xlim(-1, 33+1)
    # fig_handle.set_ylim(0, 100000)
    outname=store_path+'CSV_5_DataSet_AACode_dist.jpg'
    if IF_SaveFig==1:
        plt.savefig(outname, dpi=200)
    else:
        plt.show()
    plt.close()
    
    # -----------------------------------------------------------
    # print ("#################################")
    # print ("DICTIONARY y_data")
    # dictt=tokenizer_y.get_config()
    # print (dictt)
    # num_words = len(tokenizer_y.word_index) + 1
    # print ("################## y max token: ",num_words )
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print ("#################################")
    print ("DICTIONARY y_data: esm-", PKeys['ESM-2_Model'])
    print ("################## y max token: ",len_toks )
    
    #revere
    print ("TEST REVERSE: ")
    
#     # --------------------------------------------------------------
#     y_data_reversed=tokenizer_y.sequences_to_texts (y_data)
    
#     for iii in range (len(y_data_reversed)):
#         y_data_reversed[iii]=y_data_reversed[iii].upper().strip().replace(" ", "")
        
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # assume y_data is reversiable
    y_data_reversed = decode_many_ems_token_rec(y_data, esm_alphabet)
    
     
    print ("Element 0", y_data_reversed[0])
    print ("Number of y samples",len (y_data_reversed) )
    
    for iii in [0,2,6]:
        print("Ori and REVERSED SEQ: ", iii)
        print(seqs[iii])
        print(y_data_reversed[iii])

    # print ("Original: ", y_data[:3,:])
    # print ("REVERSED TEXT 0..2: ", y_data_reversed[0:3])
    
    print ("Len 0 as example: ", len (y_data_reversed[0]) )
    print ("CHeck ori: ", len (seqs[0]) )
    print ("Len 2 as example: ", len (y_data_reversed[2]) )
    print ("CHeck ori: ", len (seqs[2]) )
    
    # placeholder
    tokenizer_X = None
    tokenizer_Y = None
    
    return X, X_NF_List, y_data, y_data_reversed,tokenizer_X, tokenizer_Y
        
# =============================================================
# build loaders
def build_dataloaders(
    X,
    y_data,
    protein_df,
    PKeys=None,
    CKeys=None,
):
    # unload the parameters
    store_path = PKeys['data_dir']
    IF_SaveFig = CKeys['SilentRun']
    
    batch_size = PKeys['batch_size']
    TestSet_ratio = PKeys['testset_ratio']
    maxdata=PKeys['maxdata']

    
    if maxdata<y_data.shape[0]:
        print ('select subset...', maxdata )
        # X1=X1[:maxdata]
        # X2=X2[:maxdata]
        X=X[:maxdata]
        y_data=y_data[:maxdata]
        print ("new shapes (X, y_data): ", X.shape, y_data.shape)
    
    # covert into dataloder
    X_train, X_test, \
    y_train, y_test, \
    df_train, df_test \
    = train_test_split(
        X,  
        y_data,
        protein_df,
        test_size=TestSet_ratio,
        random_state=DPack_Random,
    )
    # may take a look of the train and test
    # on X
    for ii, this_X_Key in enumerate(PKeys['X_Key']):
        
        # ==========================================================
        fig = plt.figure(figsize=(24,16),dpi=200)
        fig, ax0 = plt.subplots()
        for jj in range(X_train.shape[0]):
            ax0.plot(
                X_train[jj,ii,:],
                color='b',
                alpha=0.1,
            )
        for jj in range(X_test.shape[0]):
            ax0.plot(
                X_test[jj,ii,:],
                color='r',
                alpha=0.1,
            )
        
    
        plt.xlabel('#')
        plt.ylabel('X_data')
        plt.title(f"Normalized {this_X_Key}")
        outname = store_path+f'CSV_5_Screened_NMS_{this_X_Key}_Dist.jpg'
        if IF_SaveFig==1:
            plt.savefig(outname, dpi=200)
        else:  
            plt.show()
        plt.close()
    
    
    # ++ for esm
    train_dataset = RegressionDataset(
        torch.from_numpy(X_train).float(), # already normalized
        y_train,
    )
    
    test_dataset = RegressionDataset(
        torch.from_numpy(X_test).float(), # already normalized
        y_test,
    )
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    train_loader_noshuffle = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size
    )
    
    # some test
    UPack.Print(f"In train_loader, batch #: {len(train_loader)}")
    UPack.Print(f"In test_loader, batch #: {len(test_loader)}")
    
    return train_loader, train_loader_noshuffle, test_loader, \
            df_train, df_test