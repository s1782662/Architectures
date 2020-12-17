from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch
import os


class FashionMNIST(Dataset):
    def __init__(self, dName,fName, transform=None):
        self.transform = transform
        self.df = pd.read_csv(os.path.join(dName,fName))
        self.labels = self.df.label.values
        self.images = self.df.iloc[:,1:].values.astype('uint8').reshape(-1,28,28)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        label = self.labels[idx]
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, label

