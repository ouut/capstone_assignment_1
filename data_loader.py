import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader 
import torch.nn as nn

class AugmentedFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=2), 
            transforms.ColorJitter(brightness=0.2)
        ])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # Robust conversion: np.asarray handles both Tensors and ndarrays
        img = np.asarray(img)
        img = Image.fromarray(img) 

        if self.train:
            img = self.augment(img) 

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)) 
])

train_image_data = AugmentedFashionMNIST('./data', train=True, download=True, transform=transform)
test_image_data = AugmentedFashionMNIST('./data', train=False, download=True, transform=transform)

def get_train_data_loaders(batch_size=64):
    test_loader = DataLoader(train_image_data, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader

def get_test_data_loaders(batch_size=64):
    test_loader = DataLoader(test_image_data, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader

def get_data_classes():
    return train_image_data.classes