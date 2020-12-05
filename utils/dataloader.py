import torchvision
from torchvision import transforms
import torch

aug_dataset = torchvision.datasets.ImageFolder(root='./eval_data/aug',
                                        transform=transforms.Compose([
                                        transforms.Resize((128,128)),
                                        transforms.ToTensor(),
                                        ]))

aug_loader = torch.utils.data.DataLoader(dataset=aug_dataset,
                                                batch_size=10,
                                                shuffle=True)