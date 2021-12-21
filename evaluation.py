import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import esm
# For DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix

from dataset import PPI_Dataset
from model import BaseClsModel

parser = argparse.ArgumentParser(description='PPI pretrain model method')
parser.add_argument('--test_path', default='./data_ppi/LenA400_LenB400/test_data.csv', help='data path')
parser.add_argument('--resume', default="experiments/Cls_wograd/0.ckpt", help='path to load your model')
parser.add_argument('--outdir', default='experiments/Cls_wograd', help='folder to save output')
parser.add_argument('--batchsize', default=2,type=int, help='batchsize')
# for ddp
parser.add_argument("--local_rank", default=-1, type=int)
args = parser.parse_args()
local_rank = args.local_rank
# ddp init
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
set_seed()

dataset = PPI_Dataset(args.test_path)
test_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, \
                                                num_workers=0, sampler=test_sampler)
# Model
model = BaseClsModel().to(local_rank)
ckpt_path = args.resume
# Pretrain_model
pretrain_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
pretrain_model = pretrain_model.to(local_rank)
for param in pretrain_model.parameters():
    param.requires_grad = False

# load model
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))

model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

# Specify loss function
loss_fn = nn.CrossEntropyLoss().to(local_rank)

def get_embeddings(pretrain_model, batch_tokens):
    """
       Get avg pooling of the embedding of the input sequence data
       :param batch_tokens: list[tokens]
       :return: tensor: [n,1280]
    """
    results = pretrain_model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]  # [num,maxlen,embed=1280]
    pool_embedding = token_representations.mean(1)
    return pool_embedding

def cal_confusion_matrix(y_true,y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    res = {'TP':TP/len(y_true),'FP':FP/len(y_true),'TN':TN/len(y_true),'FN':FN/len(y_true)}
    return res

def evaluate(model, val_dataloader, loss_fn):
    model.eval()
    # Tracking variables
    samples_num = 0
    val_loss = []
    y_true = []
    y_pred = []
    for idx, input_data in tqdm(enumerate(val_dataloader)):
        prot_a = input_data['prot_a']
        prot_b = input_data['prot_b']
        labels = input_data['label'].to(local_rank)
        data_1 = [(prot_a[i], input_data['seq_a'][i]) for i in range(len(prot_a))]
        data_2 = [(prot_b[i], input_data['seq_b'][i]) for i in range(len(prot_b))]
        _, _, batch_tokens1 = batch_converter(data_1)
        _, _, batch_tokens2 = batch_converter(data_2)
        batch_tokens1, batch_tokens2 = batch_tokens1.to(local_rank), batch_tokens2.to(local_rank)

        with torch.no_grad():
            u = get_embeddings(pretrain_model, batch_tokens1)
            v = get_embeddings(pretrain_model, batch_tokens2)
            features = torch.cat([u, v, torch.abs(u - v)], dim=1)
            logits = model(features)

        # Compute loss
        loss = loss_fn(logits, labels)
        val_loss.append(loss.item())
        preds = torch.argmax(logits, dim=1).flatten()
        samples_num += len(preds)
        y_true.extend(list(labels.cpu().numpy()))
        y_pred.extend(list(preds.cpu().numpy()))
    res = cal_confusion_matrix(y_true,y_pred)
    print(res)

evaluate(model,test_dataloader,loss_fn)
