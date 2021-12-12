import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

import os

# load pretrain models
import esm

class PPI_Dataset(data.Dataset):
    def __init__(self,ppi_file,maxlen = 1000):

        self.maxlen = maxlen
        self.ppi_df = pd.read_csv(ppi_file)
        self.append_toks = {'cls':'<cls>','pad':'<pad>','sep':'<sep>'}
        self.get_protidx()     # transfer protein name to idx
        self.pretrain_model, self.alphabet = esm.pretrained.esm1_t34_670M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()

    def get_protidx(self):
        cnt = 0
        all_prot = set(self.ppi_df['item_id_a'].values).union(set(self.ppi_df['item_id_b'].values))
        self.prot2idx = {prot:idx for idx,prot in enumerate(all_prot)}

    def __len__(self):
        return len(self.ppi_df)

    def __getitem__(self, idx):
        data = self.ppi_df.iloc[idx]
        seq_a = data['sequence_a']
        seq_b = data['sequence_b']
        label = data['label']
        n_pad = self.maxlen - (len(seq_a) + len(seq_b) + 2)
        input_seq = self.append_toks['cls'] + seq_a + self.append_toks['sep'] + \
                    seq_b

        ppi_name = '{}_{}'.format(data['item_id_a'],data['item_id_b'])
        data = [tuple([ppi_name, input_seq])]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        results = self.pretrain_model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]  # [num,maxlen,embed=1280]

        cls_token = token_representations[:, 0, :]
        tensor_label = torch.zeros([2],dtype=torch.long)
        if label == 1:
            tensor_label[1] == 1
        res_data = {'cls_token':cls_token,'label':tensor_label}
        return res_data

if __name__ == "__main__":
    ppi_file = './data_ppi/9606.ENSP00000217109_data.csv'
    dataset = PPI_Dataset(ppi_file)
