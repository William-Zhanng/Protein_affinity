import os

import torch
import esm
import time
# Load ESM-1b model
#model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()
batch_converter = alphabet.get_batch_converter()

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
data = [
    #("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    #("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein1and2","<cls>MKTVRQERKALTA<sep>RQQEVFDLIRDHISQTG"),
    # ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    # ("protein3",  "K A <mask> I S Q"),
]
# labels: [protein name]; batch_strs:[protein_seq],batch_tokens: tensor:[num,maxlen+2]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Extract per-residue representations (on CPU)
s = time.time()
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33] #[num,maxlen,embed=1280]
print(token_representations,token_representations.shape)
e = time.time()
print("inference time: ",e - s)
# os.makedirs('./temp',exist_ok=True)
# temp = token_representations.numpy()

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, (_, seq) in enumerate(data):
    sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0)) #take average of each amino acid
# Look at the unsupervised self-attention map contact predictions
import matplotlib.pyplot as plt
cnt = 0
for (_, seq), attention_contacts in zip(data, results["contacts"]):
    plt.matshow(attention_contacts[: len(seq), : len(seq)])
    plt.title(seq)
    # plt.show()
    plt.savefig('./{}.png'.format(cnt))
    cnt += 1