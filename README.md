## Pre-training-based protein affinity prediction

### **1. Data preprocess**
To get protein interaction Structured dataset, use preprocess.py to generate data.  

For example, your target protein is 9606.ENSP00000217109

```python
python preprocess.py --protein_idx 9606.ENSP00000217109 --out_dir data
```

You will get structured dataset in folder data, which has these columns:
* **item_id_a:** target protein A, in this case, is 9606.ENSP00000217109
* **item_id_b:** Protein B to interact with protein A.
* **sequence_a:** Amino acid sequence of protein A.
* **sequence_b:** Amino acid sequence of protein B.
* **label:** 0 or 1. 1 means affinity, 0 otherwise.


### **2. Get embedding of amino acids**
In order to get embedding vector of the protein, you can load and use a pretrained model as follows:

```python
import torch
import esm

# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein3",  "K A <mask> I S Q"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, (_, seq) in enumerate(data):
    sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
```