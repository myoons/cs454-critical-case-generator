import torch
import torchvision
from torchvision import transforms
from utils.randaugment import RandAugment
import matplotlib.pyplot as plt
import numpy as np

train_dataset = torchvision.datasets.ImageFolder(root='./data/train',
                                        transform=transforms.Compose([
                                        transforms.Resize((128,128)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        RandAugment(n=2,m=8),
                                        transforms.ToTensor(),
                                        ]))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=32,
                                                shuffle=True
                                                )

val_dataset = torchvision.datasets.ImageFolder(root='./data/eval',
                                        transform=transforms.Compose([
                                        transforms.Resize((128,128)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        RandAugment(n=2,m=8),
                                        transforms.ToTensor(),
                                        ]))

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=10,
                                                shuffle=False
                                                )

def custom_imshow(imgList, labelList):
    
    fig = plt.figure()

    rows = 2
    cols = 2

    for i in range(4):
        img = imgList[i]
        temp = fig.add_subplot(rows, cols, i+1)
        temp.imshow(np.transpose(img, (1, 2, 0)))
        temp.axis('off')
    

    plt.show()

if __name__ == "__main__":
    for batch_idx, data in enumerate(train_loader) :
        inputs, labels = data
        custom_imshow(inputs, labels)
        