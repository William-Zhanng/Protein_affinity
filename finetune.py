import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
# For DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import PPI_Dataset
from model import BaseClsModel


parser = argparse.ArgumentParser(description='PPI pretrain model method')
parser.add_argument('--datapath', default='./data_ppi/9606.ENSP00000217109_data.csv', help='data path')
parser.add_argument('--resume', default=None, help='path to load your model')
parser.add_argument('--outdir', default='experiments', help='folder to save output')
parser.add_argument('--epochs', default=10,type=str, help='epochs to train the model')
parser.add_argument('--lr', default=5e-5,type=float, help='learning rate')
parser.add_argument('--eps', default=1e-8,type=float, help='default epsilon')
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

# Data
dataset = PPI_Dataset(args.datapath)
train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, \
                                                num_workers=0, sampler=train_sampler)
# Model
model = BaseClsModel().to(local_rank)
ckpt_path = args.resume
# load model
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))

model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# Specify loss function
loss_fn = nn.CrossEntropyLoss().to(local_rank)
def train(args, model, train_dataloader, loss_fn, val_dataloader=None):
    os.makedirs(args.outdir,exist_ok=True)
    # Create the optimizer
    optimizer = AdamW(model.parameters(),lr=args.lr,eps=args.eps)
    # Total number of training steps
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)
    if dist.get_rank() == 0:
        print("Start training...\n")
    for epoch_i in range(args.epochs):
        if dist.get_rank() == 0:
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        train_dataloader.sampler.set_epoch(epoch_i)
        t0_epoch, t0_batch = time.time(), time.time()
        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()
        # For each batch of training data...
        for idx,input_data in enumerate(train_dataloader):
            batch_counts += 1
            data = input_data['cls_token']
            labels = input_data['label']
            data, labels = data.to(local_rank), labels.to(local_rank)
            optimizer.zero_grad()
            logits = model(data)
            print("idx:",idx)
            loss = loss_fn(logits, labels)
            batch_loss += loss.item()
            total_loss += loss.item()
            # Perform a backward pass to calculate gradients
            loss.backward()
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()
            # Print the loss values and time elapsed for every 20 batches
            if (idx % 2 == 0 and idx != 0) or (idx == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch
                # Print training results
                if dist.get_rank() == 0:
                    print(f"{epoch_i + 1:^7} | {idx:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        print("-" * 70)
        # Evaluation
        if not val_dataloader:
            if dist.get_rank() == 0:
                print("start evaluation!\n")
            val_loss, val_accuracy = evaluate(model, val_dataloader)
            time_elapsed = time.time() - t0_epoch
            if dist.get_rank() == 0:
                print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                print("-" * 70)
                print("\n")

        if dist.get_rank() == 0:
            model_path = os.path.join(args.outdir,"{}.ckpt".format(epoch_i))
            torch.save(model.module.state_dict(), model_path)

def evaluate(model, val_dataloader):
    model.eval()
    # Tracking variables
    correct_num = 0
    val_loss = []
    all_num = 0
    for step, input_data, labels in enumerate(val_dataloader):
        seq = input_data[1]
        data, labels = seq.to(local_rank), labels.to(local_rank)
        with torch.no_grad():
            logits = model(seq)
        # Compute loss
        loss = loss_fn(logits, labels)
        val_loss.append(loss.item())
        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        true_prediction = (preds == labels).cpu().numpy().sum()
        correct_num += true_prediction
        all_num += len(seq)

    val_loss = np.mean(val_loss)
    val_accuracy = correct_num / all_num
    return val_loss,val_accuracy

train(args,model,train_dataloader,loss_fn)