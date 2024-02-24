import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class PhaseDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class PhasePredictor(nn.Module):
    def __init__(self, input_dim, qual_input_dims, latent_dimension):
        super().__init__()
        init_qual_embeddings = []
        for qual_input_dim in qual_input_dims:
            init_qual_embeddings.append(nn.Embedding(qual_input_dim,
                                                     latent_dimension))
        self.init_qual_embeddings = nn.ModuleList(init_qual_embeddings)
        self.init_quant_embeddings = nn.Linear(input_dim - len(qual_input_dims),
                                               latent_dimension)
        embedding_layers = []
        for _ in range(2):
            embedding_layers.append(nn.Linear(latent_dimension,
                                              latent_dimension))
        self.embedding_layers = nn.ModuleList(embedding_layers)
        self.out = nn.Linear(latent_dimension, 1)


    def encode(self, x):
        x_qual_emb = torch.stack([e_i(x[:, i].long())
                                  for (i, e_i) in enumerate(self.init_qual_embeddings)], -1).sum(-1)
        x_quant_emb = self.init_quant_embeddings(x[:,
                                                   len(self.init_qual_embeddings):])

        x_emb = x_qual_emb + x_quant_emb
        x_emb = F.relu(x_emb)
        for emb in self.embedding_layers:
            x_emb = emb(x_emb)
            x_emb = F.relu(x_emb)
        out = self.out(x_emb)
        return out

    def forward(self, x):
        pred = nn.Sigmoid()(self.encode(x))
        return pred
