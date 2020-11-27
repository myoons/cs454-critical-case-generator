import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.dataloader import val_loader
from models.medium import mediumNet
from models.small import smallNet

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
PATH = './trained_model/medium_74_128px.pth'

# Our Dataset Classes
classes = ('airplane', 'cat', 'dog', 'motorbike', 'person')

model = mediumNet()
trained_weight = torch.load(PATH, map_location='cpu')
model.load_state_dict(trained_weight)

def custom_imshow(imgList, predicted):
    
    fig = plt.figure()

    rows = 2
    cols = 2

    for i in range(4):
        img = imgList[i]
        temp = fig.add_subplot(rows, cols, i+1)
        temp.set_title(classes[predicted[i]])
        temp.imshow(np.transpose(img, (1, 2, 0)))
        temp.axis('off')
    
    plt.show()

for idx, data in enumerate(val_loader):

    inputs, labels = data
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    
    custom_imshow(inputs, predicted)



