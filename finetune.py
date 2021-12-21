import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import PPI_Dataset
from model import BaseClsModel
import esm
parser = argparse.ArgumentParser(description='PPI pretrain model method')
parser.add_argument('--train_path', default='./data_ppi/LenA400_LenB400/train_data.csv', help='data path')
parser.add_argument('--test_path', default='./data_ppi/LenA400_LenB400/test_data.csv', help='data path')
parser.add_argument('--resume', default=None, help='path to load your model')
parser.add_argument('--outdir', default='experiments', help='folder to save output')
parser.add_argument('--epochs', default=10,type=str, help='epochs to train the model')
parser.add_argument('--lr', default=1e-4,type=float, help='learning rate')
parser.add_argument('--eps', default=1e-8,type=float, help='default epsilon')
parser.add_argument('--batchsize', default=1,type=int, help='batchsize')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed()
trainset = PPI_Dataset(args.train_path)
train_sampler = RandomSampler(trainset)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize,sampler=train_sampler)

# Pretrain_model
# pretrain_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
# batch_converter = alphabet.get_batch_converter()
# pretrain_model = pretrain_model.to(device)

# Model
model = BaseClsModel().to(device)
ckpt_path = args.resume
if ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))

# Specify loss function
loss_fn = nn.CrossEntropyLoss().to(device)

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

def train(args, model, train_dataloader, loss_fn, val_dataloader=None):
    os.makedirs(args.outdir,exist_ok=True)
    # Create the optimizer
    optimizer = AdamW(model.parameters(),lr=args.lr,eps=args.eps)
    # Total number of training steps
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)
    print("Start training...\n")
    for epoch_i in range(args.epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        t0_epoch, t0_batch = time.time(), time.time()
        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()
        # For each batch of training data...
        for idx,input_data in enumerate(train_dataloader):
            batch_counts += 1
            prot_a = input_data['prot_a']
            prot_b = input_data['prot_b']
            labels = input_data['label'].to(device)
            data_1 = [(prot_a[i],input_data['seq_a'][i]) for i in range(len(prot_a))]
            data_2 = [(prot_b[i], input_data['seq_b'][i]) for i in range(len(prot_b))]
            _, _, batch_tokens1 = model.batch_converter(data_1)
            _, _, batch_tokens2 = model.batch_converter(data_2)

            # _, _, batch_tokens1 = batch_converter(data_1)
            # _, _, batch_tokens2 = batch_converter(data_2)
            # batch_tokens1,batch_tokens2 = batch_tokens1.to(device),batch_tokens2.to(device)
            inputs = [batch_tokens1.to(device),batch_tokens2.to(device)]

            # with torch.no_grad():
            #     u = get_embeddings(pretrain_model,batch_tokens1)
            #     v = get_embeddings(pretrain_model,batch_tokens2)
            #     features = torch.cat([u,v,torch.abs(u-v)],dim=1)

            logits = model(inputs)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and the learning rate
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_loss += loss.item()
            total_loss += loss.item()
            # Print the loss values and time elapsed for every 20 batches
            if (idx % 50 == 0 and idx != 0) or (idx == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch
                # Print training results
                print(f"{epoch_i + 1:^7} | {idx:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        print("-" * 70)
        # Evaluation
        if not val_dataloader:
            print("start evaluation!\n")
            val_loss, val_accuracy = evaluate(model, val_dataloader)
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
            print("\n")

        model_path = os.path.join(args.outdir,"{}.ckpt".format(epoch_i))
        torch.save(model.state_dict(), model_path)

def evaluate(model, test_dataloader):
    model.eval()
    # Tracking variables
    correct_num = 0
    val_loss = []
    all_num = 0
    for idx,input_data in enumerate(test_dataloader):
        prot_a,prot_b,labels = input_data['prot_a'],input_data['prot_b'],input_data['label'].to(device)
        data_1 = [(prot_a[i], input_data['seq_a'][i]) for i in range(len(prot_a))]
        data_2 = [(prot_b[i], input_data['seq_b'][i]) for i in range(len(prot_b))]
        _, _, batch_tokens1 = model.batch_converter(data_1)
        _, _, batch_tokens2 = model.batch_converter(data_2)
        inputs = [batch_tokens1.to(device), batch_tokens2.to(device)]
        with torch.no_grad():
            logits = model(inputs)
        # Compute loss
        loss = loss_fn(logits, labels)
        val_loss.append(loss.item())
        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        true_prediction = (preds == labels).cpu().numpy().sum()
        correct_num += true_prediction
        all_num += len(data_1)

    val_loss = np.mean(val_loss)
    val_accuracy = correct_num / all_num
    return val_loss,val_accuracy

train(args,model,train_dataloader,loss_fn)