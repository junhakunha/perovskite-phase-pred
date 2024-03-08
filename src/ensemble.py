import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from src.utils.constants import DATA_DIR, DEVICE
from src.train import train_model
from src.models import PhasePredictor


def initialize_ensemble(num_models, data):
    X_MH = data["X_MH"]
    X_L = data["X_L"]
    MH_qual_num_classes = data['X_MH_qual_num_classes']

    models = []
    for i in range(num_models):
        models.append(
            PhasePredictor(
                MH_input_dim=X_MH.shape[1], 
                MH_qual_num_classes=MH_qual_num_classes, 
                latent_dimension=64, 
                L_embedding_dim=X_L.shape[1]
            )
        )
    return models

def train_ensemble(models, labelled_data, device=DEVICE):
    device = torch.device(device)

    X_MH = labelled_data["X_MH"]
    X_L = labelled_data["X_L"]
    Y = labelled_data["Y"]

    for model in models:
        model.to(device)
        train_model(model, X_MH, X_L, Y, verbose=True)

def inference(models, unlabelled_data):
    X_MH = torch.tensor(unlabelled_data['X_MH'], requires_grad=False).float()
    X_L = torch.tensor(unlabelled_data['X_L'], requires_grad=False).float()

    predictions = []
    for model in models:
        pred = model(X_MH, X_L)
        predictions.append(pred)
    
    # Get the mean and variance of the predictions
    predictions = torch.stack(predictions)
    mean = predictions.mean(dim=0)
    var = predictions.var(dim=0)

    return mean, var


def acquisition_func(mean, std, xi):
    return mean + xi * std

def get_best_k_reactions(dataset, acquisition, k):
    indices = np.argsort(acquisition.cpu().detach().numpy().flatten())
    reactions = []
    for reaction in dataset['reaction_ids'][indices[-k:]]:
        reactions.append(reaction)
    return reactions

def get_worst_k_reactions(dataset, acquisition, k):
    indices = np.argsort(acquisition.cpu().detach().numpy().flatten())
    reactions = []
    for reaction in dataset['reaction_ids'][indices[:k]]:
        reactions.append(reaction)
    return reactions

def show_histogram(data, name=""):
    plt.hist(data.cpu().detach().numpy(), bins=100)
    plt.title(name)
    plt.show()

def main(num_models=10, xi=1):
    labelled_data = np.load(os.path.join(DATA_DIR, "labelled_dataset.npz"))
    unlabelled_data = np.load(os.path.join(DATA_DIR, "unlabelled_dataset.npz"))

    models = initialize_ensemble(num_models, labelled_data)
    train_ensemble(models, labelled_data)

    mean, var = inference(models, unlabelled_data)
    std = torch.sqrt(var)
    
    show_histogram(mean, "Mean")
    show_histogram(std, "Standard Deviation")

    acquisition = acquisition_func(mean, std, xi)
    best_reactions = get_best_k_reactions(unlabelled_data, acquisition, 10)
    worst_reactions = get_worst_k_reactions(unlabelled_data, acquisition, 10)

    print("Best reactions: ")
    for reaction in best_reactions:
        print(reaction)
    
    print("="*50)

    print("Worst reactions: ")
    for reaction in worst_reactions:
        print(reaction)


if __name__ == "__main__":
    main(num_models=10, xi=1)