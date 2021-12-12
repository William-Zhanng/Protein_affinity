import torch
import torch.nn as nn
import esm

class BaseClsModel(nn.Module):
    def __init__(self, n_embedding=1280, n_hidden=50, n_classes=2):
        super(BaseClsModel, self).__init__()
        self.model_name = 'BaseClsModel'
        self.n_embedding = n_embedding
        self.n_classes = n_classes
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding, n_hidden),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(n_hidden, n_classes)
        )

    def forward(self,data):
        out = self.classifier(data)
        out = torch.nn.functional.sigmoid(out)
        return out


