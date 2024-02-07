import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np



class Dataset:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize((0.5,), (0.5,)) 
        ])


        self.train_data = datasets.MNIST(
            root='data', 
            train=True, 
            download=True, 
            transform=self.transform
        )


        self.test_data = datasets.MNIST(
            root='data', 
            train=False, 
            download=True, 
            transform=self.transform
        )


