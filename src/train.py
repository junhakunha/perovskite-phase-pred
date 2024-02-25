import os
import sys
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import argparse

sys.path.append("../")
sys.path.append(os.getcwd())

from src.models import PhasePredictor
from src.datasets import PhaseDataset
from src.utils.constants import HOME_DIR, DATA_DIR, DEVICE, SEED
from src.utils.data_utils import create_confusion_matrix


def create_dataloaders(X, Y, train_ratio=0.8, seed=SEED, device="cpu"):
    X = torch.tensor(X, device=device).float()
    Y = torch.tensor(Y, device=device).float()

    num_data_points = X.shape[0]

    np.random.seed(seed)
    random_indices = np.random.permutation(num_data_points)
    num_train = int(num_data_points*train_ratio)

    train_dataset = PhaseDataset(X[random_indices[0:num_train]],
                                Y[random_indices[0:num_train]])
    test_dataset = PhaseDataset(X[random_indices[num_train:]],
                                Y[random_indices[num_train:]])

    class_sample_count = torch.tensor([len(torch.where(train_dataset.Y == t)[0])
                                    for t in np.unique(train_dataset.Y)])
    weight = 1. / class_sample_count
    samples_weight = torch.tensor([weight[t.long()] for t in train_dataset.Y])

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=16,
                                  sampler=sampler)

    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=16,
                                 shuffle=False)
    
    return train_dataloader, test_dataloader


def train(model, X, Y, seed=SEED, train_ratio=0.8, num_epochs=400, lr=3e-4, device="cpu", conf_mat=False):
    train_dataloader, test_dataloader = create_dataloaders(X, Y, train_ratio=train_ratio, seed=seed, device=device)
    num_train_batches = len(train_dataloader)
    num_test_batches = len(test_dataloader)

    opt = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, verbose=True)
    criterion = torch.nn.BCELoss()

    writer = SummaryWriter(log_dir=os.path.join(HOME_DIR, "src/logs"))
    for epoch in range(num_epochs):
        epoch_train_loss = 0
        for (train_x, train_y) in train_dataloader:
            opt.zero_grad()

            pred_y = model(train_x).flatten()
            
            loss = criterion(pred_y, train_y)
            loss.backward()
            epoch_train_loss += loss.item()

            opt.step()
        
        epoch_test_loss = 0
        for (test_x, test_y) in test_dataloader:
            pred_y = model(test_x).flatten()
            loss = criterion(pred_y, test_y)
            epoch_test_loss += loss.item()
        epoch_train_loss /= num_train_batches
        epoch_test_loss /= num_test_batches

        writer.add_scalar("Loss/train", epoch_train_loss, epoch)
        writer.add_scalar("Loss/test", epoch_test_loss, epoch)
        scheduler.step(epoch_test_loss)

    if conf_mat:
        create_confusion_matrix(model, test_dataloader, save_path=None)


def main(args):
    device = torch.device(DEVICE)
    data_path = args.data_path
    data = np.load(data_path)

    X = data["X"]
    Y = data["Y"]
    qual_input_dims = data['X_qual_num_classes']

    model = PhasePredictor(input_dim=X.shape[1], 
                           qual_input_dims=qual_input_dims, 
                           latent_dimension=64)
    model.to(device)

    train(model, 
          X, 
          Y, 
          seed=args.seed, 
          train_ratio=args.train_ratio, 
          num_epochs=args.num_epochs, 
          lr=args.lr, 
          device=device,
          conf_mat=args.conf_mat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a PhasePredictor model on the dataset.")

    dataset_path = os.path.join(DATA_DIR, "dataset.npz")

    parser.add_argument("--data_path", 
                        type=str, 
                        default=dataset_path, 
                        help="Path to npz file containing the full dataset (attempted reactions).")
    
    parser.add_argument("--train_ratio",
                        type=float,
                        default=0.8,
                        help="Ratio of train:test data (default 0.8, no val set).")
    
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for reproducibility (default 42).")
    
    parser.add_argument("--num_epochs",
                        type=int,
                        default=400,
                        help="Number of epochs to train for (default 400).")
    
    parser.add_argument("--lr",
                        type=float,
                        default=3e-4,
                        help="Learning rate for training (default 3e-4).")
    
    parser.add_argument("--conf_mat",
                        type=bool,
                        default=False,
                        help="Create a confusion matrix after training. Modify code to save to file.")
 
    args = parser.parse_args()

    main(args)



    