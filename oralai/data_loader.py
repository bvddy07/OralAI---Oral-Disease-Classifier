import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .config import TRAIN_PATH, TEST_PATH, IMG_SIZE, BATCH_SIZE

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
    test_dataset = datasets.ImageFolder(root=TEST_PATH, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset
