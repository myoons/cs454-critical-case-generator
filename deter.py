import imgaug.augmenters as iaa
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

iaa.Add(40)
iaa.Add(-40)
iaa.Multiply(0.7)
iaa.Multiply(1.5)
iaa.Cutout(fill_mode="constant", cval=(0, 255), fill_per_channel=1)
iaa.SaltAndPepper(0.05)
iaa.GaussianBlur(1.5)
iaa.MotionBlur(k=15, angle=60, direction=1)
iaa.MotionBlur(k=5, angle=60, direction=-1)
iaa.Grayscale(0.5)
iaa.SigmoidContrast(gain=10, cutoff=0.3)
iaa.LogContrast(0.7)
iaa.LogContrast(1.3)
iaa.Sharpen(alpha=0.5, lightness=0.8)
iaa.Sharpen(alpha=0.5, lightness=1.2)
iaa.Fliplr(1)
iaa.Flipud(1)
iaa.Rotate(-60)
iaa.Rotate(60) 
iaa.ShearX(-20)
iaa.ShearX(20)
iaa.ShearY(-20)
iaa.ShearY(20)
iaa.ScaleX(0.5)
iaa.ScaleX(1.5)
iaa.ScaleY(0.5)
iaa.ScaleY(1.5)


augList = [iaa.Add((-40, 40)),
           iaa.Multiply((0.5, 1.5)),
            iaa.Cutout(fill_mode="constant", cval=(0, 255), fill_per_channel=1), 
            iaa.SaltAndPepper(0.1), 
            iaa.GaussianBlur(sigma=(0.0, 3.0)), 
            iaa.MotionBlur(k=15, angle=[-45, 45]), 
            iaa.Grayscale(alpha=(0.0, 1.0)), 
            iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)), 
            iaa.LogContrast(gain=(0.6, 1.4)), 
            iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)), 
            iaa.Fliplr(1),
            iaa.Flipud(1),
            iaa.Rotate((-45, 45)), 
            iaa.ShearX((-20, 20)),
            iaa.ShearX((-20, 20)), 
            iaa.ScaleX((0.5, 1.5)), 
            iaa.ScaleY((0.5, 1.5)), 
            ]

img = np.array(Image.open('eval_data/aug/cat/cat_0307.jpg'))
aug = iaa.Multiply(0.7)
add_dark = iaa.Add(-40)

plt.imshow(iaa.Sharpen(alpha=0, lightness=0.8).augment_image(img))


plt.show()