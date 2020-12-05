
import numpy as np
import imgaug.augmenters as iaa
from ML.utils.dataloader import train_loader
import cv2
import random 

import matplotlib.pyplot as plt

critical_1 = [
    iaa.SaltAndPepper(0.05),
    iaa.Sharpen(alpha=0.2, lightness=0.9),
    iaa.ScaleX(0.7)
]

critical_2 = [
    iaa.SaltAndPepper(0.05),
    iaa.Sharpen(alpha=0.2, lightness=0.9),
    iaa.Flipud(1)
]

critical_3 =[
    iaa.SaltAndPepper(0.05),
    iaa.Sharpen(alpha=0.2, lightness=0.9),
    iaa.ShearY(10)
]

critical_4 =[
    iaa.SaltAndPepper(0.05),
    iaa.Sharpen(alpha=0.2, lightness=0.9),
    iaa.Add(20)
]

critical_5 =[
    iaa.SaltAndPepper(0.05),
    iaa.Sharpen(alpha=0.2, lightness=0.9),
    iaa.Multiply(1.3)
]

critical_6 =[
    iaa.SaltAndPepper(0.05),
    iaa.Sharpen(alpha=0.2, lightness=0.9),
    iaa.LogContrast(1.3)
]

critical_cases = [critical_1, critical_2, critical_3, critical_4, critical_5, critical_6]

# Our Dataset Classes
classes = ('airplane', 'cat', 'dog', 'motorbike', 'person')

def ImgTransform(images, TransformList):
    
    seq = iaa.Sequential([
        TransformList[0],
        TransformList[1],
        TransformList[2],
    ])

    I = cv2.normalize(images.permute(0,2,3,1).numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) # 10, 128, 128, 3
    result = seq(images=I)
    return np.array(result, dtype='float32') / 255 # batchSize , 128, 128, 3


for idx, data in enumerate(train_loader):

    inputs, labels = data[0], data[1] # Batchsize : 30
    rand_aug = random.randint(0,5)

    aug_im = ImgTransform(inputs, critical_cases[rand_aug])
    for img_idx, img in enumerate(aug_im) :

        # plt.imshow(img)
        # plt.show()

        plt.imsave('created/{}/{}_{}.png'.format(classes[labels[img_idx]], idx, img_idx), img)

        