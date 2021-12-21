import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import time
import os

# load pretrain models
import esm

class PPI_Dataset(data.Dataset):
    def __init__(self,ppi_file,maxlen = 600):

        self.maxlen = maxlen
        self.ppi_df = pd.read_csv(ppi_file)
        self.append_toks = {'cls':'<cls>','pad':'<pad>','sep':'<sep>'}
        self.get_protidx()     # transfer protein name to idx
        # self.pretrain_model, self.alphabet = esm.pretrained.esm1_t34_670M_UR50S()
        # self.batch_converter = self.alphabet.get_batch_converter()

    def get_protidx(self):
        cnt = 0
        all_prot = set(self.ppi_df['item_id_a'].values).union(set(self.ppi_df['item_id_b'].values))
        self.prot2idx = {prot:idx for idx,prot in enumerate(all_prot)}

    def __len__(self):
        return len(self.ppi_df)

    # def __getitem__(self, idx):
    #     # <cls> + seq1 + <sep> + seq2
    #     data = self.ppi_df.iloc[idx]
    #     seq_a = data['sequence_a']
    #     seq_b = data['sequence_b']
    #     label = data['label']
    #     n_pad = self.maxlen - (len(seq_a) + len(seq_b) + 2)
    #     input_seq = self.append_toks['cls'] + seq_a + self.append_toks['sep'] + \
    #                 seq_b
    #     input_seq = input_seq[:self.maxlen]
    #
    #     ppi_name = '{}_{}'.format(data['item_id_a'],data['item_id_b'])
    #     # tensor_label = torch.zeros([2],dtype=torch.long)
    #     # if label == 1:
    #     #     tensor_label[1] == 1
    #     tensor_label = torch.tensor(label,dtype=torch.long)
    #     input_data = {'ppi_name':ppi_name,'seq':input_seq,'label':tensor_label}
    #     return input_data
    def __getitem__(self, idx):
        data_item = self.ppi_df.iloc[idx]
        label = data_item['label']
        # No need to padding, because use pooling, and preprocessing filter the protein sequence length > 700
        tensor_label = torch.tensor(label, dtype=torch.long)
        # data_item['sequence_a'] += (self.maxlen- len(data_item['sequence_a']))*self.append_toks['pad']
        # data_item['sequence_b'] += (self.maxlen- len(data_item['sequence_b'])) * self.append_toks['pad']
        input_data = {'prot_a':data_item['item_id_a'],'prot_b':data_item['item_id_b'],'seq_a':data_item['sequence_a'],
                      'seq_b':data_item['sequence_b'],'label':tensor_label}

        return input_data


if __name__ == "__main__":
    ppi_file = './data_ppi/9606.ENSP00000217109_data.csv'
    dataset = PPI_Dataset(ppi_file)
