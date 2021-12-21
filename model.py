import torch
import torch.nn as nn
import esm

class BaseClsModel(nn.Module):
    def __init__(self, n_embedding=1280, n_hidden=50, n_classes=2):
        super(BaseClsModel, self).__init__()
        self.model_name = 'BaseClsModel'
        self.n_embedding = n_embedding
        self.n_classes = n_classes
        # pretrain model
        # self.pretrain_model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        # self.pretrain_model, self.alphabet = esm.pretrained.esm1_t12_85M_UR50S()
        # self.batch_converter = self.alphabet.get_batch_converter()

        self.classifier = nn.Sequential(
            nn.Linear(n_embedding*3, n_hidden),
            nn.ReLU(),
#            nn.Dropout(0.5),
            nn.Linear(n_hidden, n_classes)
        )

    def get_embeddings(self,batch_tokens):
        """
           Get avg pooling of the embedding of the input sequence data
           :param batch_tokens: list[tokens]
           :return: tensor: [n,1280]
        """
        # prot_reprs = []
        # for tokens in batch_tokens:
        #     results = self.pretrain_model(tokens.unsqueeze(0), repr_layers=[33], return_contacts=True)
        #     token_representations = results["representations"][33]  # [num,maxlen,embed=1280]
        #     pool_embedding = token_representations.mean(1)
        #     prot_reprs.append(pool_embedding)
        # representations = torch.cat(prot_reprs,1)
        results = self.pretrain_model(batch_tokens, repr_layers=[12], return_contacts=True)
        token_representations = results["representations"][12]  # [num,maxlen,embed=1280]
        pool_embedding = token_representations.mean(1)
        return pool_embedding

    def forward(self,data):
        # u = self.get_embeddings(data[0])  # pooling features of protein A
        # v = self.get_embeddings(data[1])  # pooling features of protein B
        # features = torch.cat([u,v,torch.abs(u-v)],dim=1)  # concated features of [u,v,|u-v|]
        # out = self.classifier(features)

        out = self.classifier(data)
        out = torch.sigmoid(out)
        return out


