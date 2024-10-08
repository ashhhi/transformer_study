import os.path
import random

import numpy as np
import torch
from keras.utils import to_categorical
from torch.utils.data import Dataset
from torch import nn
from PIL import Image
from torchvision.transforms import v2
from sklearn.preprocessing import OneHotEncoder

class MyDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.transform = transform
        self.image_list = []
        for _, _, filelist in os.walk(root_path):
            for name in filelist:
                if name.lower().endswith(('.jpg', '.png')):
                    self.image_list.append(os.path.join(root_path, name))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image


if __name__ == '__main__':
    transforms = v2.Compose([
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
        v2.Resize((320, 224), antialias=False),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(MySegmentationDataset('/Users/shijunshen/Documents/Code/dataset/Segmentation', transforms).__getitem__(2))
