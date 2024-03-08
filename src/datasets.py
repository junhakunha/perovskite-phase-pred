from torch.utils.data import Dataset


class PhaseDataset(Dataset):
    def __init__(self, X_MH, X_L, Y):
        self.X_MH = X_MH
        self.X_L = X_L
        self.Y = Y

    def __len__(self):
        return self.X_MH.shape[0]

    def __getitem__(self, index):
        return self.X_MH[index], self.X_L[index], self.Y[index]