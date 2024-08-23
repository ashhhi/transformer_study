from torchvision import datasets
import torch
import os
from torchvision import transforms


transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset_train = datasets.ImageFolder('/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Final-Dataset', transform=transform_train)
print(dataset_train)
data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=2,
        num_workers=0,
        drop_last=True,
)

print(data_loader_train)
for data in data_loader_train:
    print(data)