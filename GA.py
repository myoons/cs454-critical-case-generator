import torchvision
import numpy as np
import os
import torch
import random
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

import cv2
from utils.dataloader import aug_loader

from ML.models.medium import mediumNet
from ML.models.small import smallNet

classes = ('airplane', 'cat', 'dog', 'motorbike', 'person')

def prepare_model():

    print('Preparing Model..')
    
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    PATH = './ML/trained_model/medium_74_128px.pth'
    model = mediumNet()
    trained_weight = torch.load(PATH, map_location='cpu')
    model.load_state_dict(trained_weight)

    return model

augList = [iaa.Add((-40, 40)),
           iaa.Multiply((0.5, 1.5)),
            iaa.Cutout(fill_mode="constant", cval=(0, 255), fill_per_channel=1), 
            iaa.SaltAndPepper(0.1),
            # iaa.Cartoon(), 
            # iaa.BlendAlphaRegularGrid(nb_rows=(4, 6), nb_cols=(1, 4)), 
            iaa.GaussianBlur(sigma=(0.0, 3.0)), 
            iaa.MotionBlur(k=15, angle=[-45, 45]), 
            iaa.Grayscale(alpha=(0.0, 1.0)), 
            # iaa.MultiplySaturation((0.5, 1.5)), 
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
            #iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03)), 
            #iaa.pillike.EnhanceSharpness()
            ]

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

def imshow(imgList):
    
    fig = plt.figure()

    rows = 2
    cols = 2

    for i in range(4):
        img = imgList[i]
        temp = fig.add_subplot(rows, cols, i+1)
        temp.imshow(np.transpose(img, (1, 2, 0)))
        temp.axis('off')
    
    plt.show()

# Our Dataset Classes
classes = ('airplane', 'cat', 'dog', 'motorbike', 'person')

# 각 aug text에 맞는 함수 실행 매칭 필요
# fitness fuction 업뎃 필요 = label_fit
def aug_GA(label, augList, popN, genN, rate, target_score, model):
    
    # gen0 : 제일 처음 population
    # 기본적으로 gen list의 구조는 [ [augComb1, label_fit1],[augComb2, label_fit2], ... ]
    # [0.1, 0.3, 0.5, 0.1, 0.1]

    gen0 = []
    for n in range(popN):
        randAug = make_augComb(augList,4)
        augFit = label_fit(label, randAug, model)
        gen0.append([randAug, augFit])
    
    gen = gen0
    gen_num = 0
    status = True
    while status:
        new_gen = GA(label, augList, gen, genN, rate, model)
        for son in new_gen:
            if son[1] > target_score:
                status = False
                finAug, finFit = son[0], son[1]
            #print(son[1])
        print('Generation : {}'.format(gen))
        gen = new_gen
        gen_num += 1
        #print(gen_num)

    return finAug, finFit


# augList에서 num(4)개만큼 augmentation 골라 리스트로 반환
def make_augComb(augList, num):
    augComb = random.sample(augList,num)
    return augComb

# = score (크리티컬한 케이스일수록 값이 커짐 => 값이 크면 성능이 안 좋지만 우리가 찾아야 할 것)
# label과 생성된 랜덤 augs를 받아 aug를 적용시킨 것의 fit값을 불러옴
# augs = [aug_name1, name2, name3, name4]

#################

# fitness.py
# Input: 4가지 이미지 변형 조합 리스트, 맞는 라벨 (Original Label)
# Output: fitness

def ImgTransform(images, TransformList):

    seq = iaa.Sequential([
        TransformList[0],
        TransformList[1],
        TransformList[2],
        TransformList[3]
    ])

    I = cv2.normalize(images.permute(0,2,3,1).numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) # 10, 128, 128, 3
    result = seq(images=I)
    return torch.tensor(np.array(result, dtype='float32') / 255).permute(0,3,1,2) # 창준 여기 잠들다

def label_fit(labelIdx, augList, model):
    fitnessList = []

    for idx, data in enumerate(aug_loader):
        
        inputs, labels = data # Inputs : image 10 개 / labels : airplane -> cat -> dog -> motorbike -> person 순
        labelIdx = labels[0]
        # get an augmented images from ImgTransform
        aug_im = ImgTransform(inputs, augList) # -> 창준씨 파트로 연결 
        
        # put images into ML and get the result
        with torch.no_grad():
            outputs = model(aug_im) # ML 인풋 아웃풋 형태 맞게 바꿔야 함
            resList = torch.softmax(outputs, dim=-1).tolist()
            _, predicted = torch.max(outputs.data, 1)

        for i in range(10):

            first = sorted(resList[i])[4]
            firstIdx = sorted(range(len(resList[i])), key=lambda k: resList[i][k])[4]
            second = sorted(resList[i])[3]
            secondIdx = sorted(range(len(resList[i])), key=lambda k: resList[i][k])[3]

            var = np.var(sorted(resList)[0:4])
            # calculate the fitness for each case

            f = (1-resList[i][labelIdx] + 0.5*(1-first+second) + var) * np.exp(first-resList[i][labelIdx])
            fitnessList.append(f)

            # [Tensor]

    # calculate the total fitness as average of 10 fitnesses
    fitnessTotal = np.mean(fitnessList)
    return fitnessTotal

########

# gen list를 받아서 룰렛-휠 방식에 따라 2개의 스코어가 높은 aug부모를 픽함
def roulette(gen):
    augs = [g[0] for g in gen]
    scores = [g[1] for g in gen]
    relaScore = [f/sum(scores) for f in scores]
    pick = random.choices(augs, weights = relaScore, k = 2)
    return pick[0], pick[1]

def crossover(augList, a, b):
    i = random.randrange(len(a))
    ai = a[:i+1]
    bi = b[i+1:]
    augListName = [g for g in augList]
    aiName = [m[0] for m in ai]
    biName = [n[0] for n in bi]
    possible = list(set(augListName).difference(set(aiName).union(set(biName))))
    change = list(set(aiName).intersection(set(biName)))

    biNew = []
    for k in bi:
        if k in change:
            name = random.choice(possible)
            for c in range(len(augList)):
                if name == augList[c]:
                    biNew.append(augList[c])
        else:
            biNew.append(k)
    
    return ai + biNew

# crossover 결과물 augs C를 일정 확률 안에서 변이를 일으킴
def mutate(augList, C ,rate):
    poss = list(set(augList).difference(set(C)))
    for ch in range(len(C)):
        if(random.random() < rate):
            C[ch] = random.choice(poss)
    return C

def GA(label, augList, gen, genN, rate, model):
    new_gen = []
    for i in range(genN):
        A, B = roulette(gen)
        C = crossover(augList, A, B)
        Cn = mutate(augList, C, rate)
        new_gen.append([Cn, label_fit(label, Cn, model)]) # [iaa]
    sortGen = sorted(gen, key = lambda x : x[1])
    new_gen = new_gen + sortGen[genN]
    return new_gen
    

if __name__ == "__main__":
    model = prepare_model()
    fin_aug, fin_fit = aug_GA(0, augList, 50, 40, 0.5, 1, model)
    print(fin_aug, fin_fit)