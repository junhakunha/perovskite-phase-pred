import torch
import torch.nn as nn
import torch.nn.functional as F
import rdkit.Chem

class PhasePredictor(nn.Module):
    def __init__(self, MH_input_dim, MH_qual_num_classes, latent_dimension, L_embedding_dim):
        """
        Initialize the phase predictor model.

        Args:
        - MH_input_dim: int, the number of features in the Metal, Halide input (qualitative and quantitative)
                        Note qualitative features always come first in the input.
        - MH_qual_num_classes: list of integers, the number of classes for each of the qualitative features of Metal, Halide input
        - latent_dimension: int, the dimension of the latent space
        - L_embedding_dim: int, the dimension of the Ligand embeddings. Used to match the dimensions of the Metal, Halide embeddings.
        """
        super().__init__()
        # Set up embeddings for the qualitative features of Metal, Halide input (categorical)
        MH_qual_embeddings = []
        for ML_qual_num_class in MH_qual_num_classes:
            # Match to the dimensions of Ligand embeddings
            MH_qual_embeddings.append(nn.Embedding(ML_qual_num_class, L_embedding_dim//2))
        self.MH_qual_embeddings = nn.ModuleList(MH_qual_embeddings)

        # Set up embeddings for the quantitative features of Metal, Halide input (continuous)
        MH_num_quant_features = MH_input_dim - len(MH_qual_num_classes)
        self.MH_quant_embeddings = nn.Linear(MH_num_quant_features, L_embedding_dim//2)

        # Fusion layers to combine the embeddings of Metal, Halide embeddings with Ligand embeddings
        fusion_layers = [
            nn.Linear(2*L_embedding_dim, latent_dimension),
            nn.Linear(latent_dimension, latent_dimension)
        ]
        self.fusion_layers = nn.ModuleList(fusion_layers)

        # Prediction layer to predict the phase
        self.prediction_layer = nn.Linear(latent_dimension, 1)

    def encode_MH(self, x_MH):
        """
        Encode the Metal, Halide input. x_MH has the features for Metal, Halide input, with qualitative features first.
        """
        # Sum all embeddings for qualitative features
        x_MH_qual_emb = torch.stack(
            [e_i(x_MH[:, i].long()) for (i, e_i) in enumerate(self.MH_qual_embeddings)], 
            -1
        ).sum(-1)

        # Sum all embeddings for quantitative features
        x_MH_quant_emb = self.MH_quant_embeddings(x_MH[:, len(self.MH_qual_embeddings):])
        x_MH = torch.cat((x_MH_qual_emb, x_MH_quant_emb), -1)
        return x_MH

    def forward(self, x_MH, x_L):
        x_MH = self.encode_MH(x_MH)
        x = torch.cat([x_MH, x_L], -1)

        x = F.relu(x)
        for layer in self.fusion_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.prediction_layer(x)
        pred = F.sigmoid(x)
        
        return pred