# fitness.py
# Input: 4가지 이미지 변형 조합 리스트, 맞는 라벨 (Original Label)
# Output: fitness

'''
이미지 데이터셋에서 Original Label에 해당하는 이미지 10개 가져오기
ImgTransform에 돌려서 변형된 이미지 10개 만들고
그걸 다시 ML에 넣어서 라벨링한 결과 받아서 fitness 계산하고 평균내기
-> 원래 라벨과 비교하여 fitness 계산 (이거 식 정하기)
'''

import numpy as np
import imgaug as ia
from PIL import Image


arr = [1,2,3]
'''
np.mean(arr) #mean
np.var(arr) #variation 분산
np.std(arr) #standard deviation
'''



def fitness(augList, labelIdx):
    labels = [‘airplane’, ‘cat’, ‘dog’, ‘motorbike’, ‘person’]
    fitnessList = []

    for i in range (10) :
        # get an image from data set
        im = Image.open('testImage.jpeg') # 어느 폴더에서 어떻게 불러올지 정해야함


        # get an augmented images from ImgTransform
        aug_im = ImgTransform(im, augList) # -> 창준씨 파트로 연결 


        # put images into ML and get the result
        resList = ML(aug_im) # ML 인풋 아웃풋 형태 맞게 바꿔야 함
        first = sorted(resList)[0]
        firstIdx = sorted(range(len(resList)), key=lambda k: s[k])[0]
        second = sorted(resList)[1]
        secondIdx = sorted(range(len(resList)), key=lambda k: s[k])[1]


        # calculate the fitness for each case

        f = (1-resList[labelIdx]) + 1 / (10 * (first-second)) # 더 잘 만들어볼 필요 있음 
        fitnessList.append(f)


    # calculate the total fitness as average of 10 fitnesses
    fitnessTotal = np.mean(fitnessList)
    return fitnessTotal



print('hi')
print(fitness(arr, 0))
