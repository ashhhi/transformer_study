import os
import cv2 as cv
import numpy as np
from tqdm import tqdm

path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Final-Dataset/self-supervision'

m_all = [0, 0, 0]
s_all = [0, 0, 0]

for _, _, filename in os.walk(path):
    # print(len(filename))
    for file in tqdm(filename):
        if file.lower().endswith(('.png','.jpg')):
            img_path = os.path.join(path, file)
            img = cv.imread(img_path)
            img = img.reshape((-1, 3))
            pixel_num = img.shape[0] * img.shape[1]
            for i in range(3):
                m = np.mean(img, axis=0)
                v = np.std(img, axis=0)
                m_all += m
                s_all += v
                print(m_all, s_all)
    m_all = m_all / len(filename)
    s_all = s_all / len(filename)

print(m_all, s_all)
