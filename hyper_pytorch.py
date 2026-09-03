import torch
from torch.utils.data import Dataset

class HyperData(Dataset):

    def __init__(self, dataset, transform=None):

        self.data = dataset[0]
        self.labels = dataset[1]
        self.transform = transform

    def __getitem__(self, index):

        x = self.data[index]
        y = self.labels[index]

        x = torch.tensor(
            x,
            dtype=torch.float32
        )

        y = torch.tensor(
            y,
            dtype=torch.long
        )

        return x, y

    def __len__(self):

        return len(self.labels)
        return len(self.labels)