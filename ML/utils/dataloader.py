import torch
import torchvision
from torchvision import transforms
from utils.randaugment import RandAugment
import matplotlib.pyplot as plt
import numpy as np

train_dataset = torchvision.datasets.ImageFolder(root='./data/train',
                                        transform=transforms.Compose([
                                        transforms.Resize((128,128)),
                                        # transforms.RandomHorizontalFlip(),
                                        # transforms.RandomVerticalFlip(),
                                        # RandAugment(n=2,m=8),
                                        transforms.ToTensor(),
                                        ]))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=32,
                                                shuffle=True
                                                )

val_dataset = torchvision.datasets.ImageFolder(root='./data/eval',
                                        transform=transforms.Compose([
                                        transforms.Resize((128,128)),
                                        # transforms.RandomHorizontalFlip(),
                                        # transforms.RandomVerticalFlip(),
                                        # RandAugment(n=2,m=8),
                                        transforms.ToTensor(),
                                        ]))

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=10,
                                                shuffle=True
                                                )