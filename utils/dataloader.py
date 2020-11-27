import torchvision
from torchvision import transforms
import torch
from ML.utils.randaugment import RandAugment

aug_dataset = torchvision.datasets.ImageFolder(root='./data/aug',
                                        transform=transforms.Compose([
                                        transforms.Resize((128,128)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        RandAugment(n=2,m=8),
                                        transforms.ToTensor(),
                                        ]))

aug_loader = torch.utils.data.DataLoader(dataset=aug_dataset,
                                                batch_size=10,
                                                shuffle=False)