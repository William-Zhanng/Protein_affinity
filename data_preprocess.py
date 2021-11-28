import pandas as pd
import numpy as np
import tqdm
import argparse

import random
import os

def create_data(protein_idx,all_data,pseq_dict,out_path,sample_ratio=100):
    
    # Postive sample
    items = all_data[all_data['item_id_a'] == protein_idx]
    df_out = pd.DataFrame(columns = ['item_id_a','item_id_b','sequence_a','sequence_b','label']) 
    target_proteins = pd.Series(items['item_id_b'].values)
    df_out['item_id_b'] = target_proteins
    df_out['item_id_a'] = protein_idx
    df_out['label'] = 1
    df_out['sequence_a'] = pseq_dict[protein_idx]
    seq_list = []
    for i in range(df_out.shape[0]):
        target = target_proteins[i]
        target_seq = pseq_dict[target]
        seq_list.append(target_seq)
    seq_list = pd.Series(seq_list)
    df_out['sequence_b'] = seq_list
    
    # Negative sample
    # Neg sample compute
    all_idx = set(all_data['item_id_a'])
    target_idx = set(df_out['item_id_b'])
    neg_idx = all_idx - target_idx
    sample_ratio = 100
    sample_num = sample_ratio * len(target_idx)
    neg_prot = random.sample(neg_idx,min(len(neg_idx),sample_num))
    
    # Create neg sample dataframe
    df_neg = pd.DataFrame(columns = ['item_id_a','item_id_b','sequence_a','sequence_b','label'])
    df_neg['item_id_b'] = pd.Series(neg_prot)
    df_neg['item_id_a'] = protein_idx
    df_neg['label'] = 0
    df_neg['sequence_a'] = pseq_dict[protein_idx]
    seq_list = []
    for i in range(df_neg.shape[0]):
        target = neg_prot[i]
        target_seq = pseq_dict[target]
        seq_list.append(target_seq)
    seq_list = pd.Series(seq_list)
    df_neg['sequence_b'] = seq_list
    
    df_out = pd.concat([df_out,df_neg],axis=0)
    df_out.to_csv(out_path)
    return df_out


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--actions_data_path', type=str, default='./data/9606.protein.actions.all_connected.txt')
    parser.add_argument('--pseq_path', type=str, default='./data/protein.STRING_all_connected.sequences.dictionary.tsv',
                        help='The protein sequence data file path')
    parser.add_argument('--protein_idx', type=str, default='9606.ENSP00000217109',
                        help='The name of protein in STRING dataset')
    parser.add_argument('--out_dir', type=str, default='./data',
                        help='The protein sequence data file path')                    
    args = parser.parse_args()

    os.makedirs(args.out_dir,exist_ok=True)
    all_data = pd.read_csv(args.actions_data_path,sep='\t')
    # Create protein sequence dict
    pseq_dict = {}
    for line in open(args.pseq_path):
        line = line.strip().split('\t')
        pseq_dict[line[0]] = line[1]

    all_data.drop(columns=['mode','action','is_directional','a_is_acting','score'],axis=1,inplace=True)
    all_data.drop_duplicates(inplace = True)
    df_each_cnt = all_data.groupby('item_id_a').count()['item_id_b'].sort_values()
    protein_idxs = list(df_each_cnt[(df_each_cnt >= 150) & (df_each_cnt < 300)].index)    
    df_out = create_data(args.protein_idx,all_data,pseq_dict,os.path.join(args.out_dir,'{}_data.csv'.format(args.protein_idx)))