import numpy as np
import torchvision, torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from vision_transformer import VisionTransformer



train_data = torchvision.datasets.CIFAR10("./dataset", train=True, download=True, transform= torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform= torchvision.transforms.ToTensor())


train_dataloader = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=0, drop_last=False)
test_dataloader = DataLoader(dataset=test_data, batch_size=128, shuffle=True, num_workers=0, drop_last=False)



ViT = VisionTransformer(4, 8, 8, 10, device='mps').to(device='mps')

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(ViT.parameters(), lr=1e-2)

# 开始训练
total_train_step = 0
total_test_step = 0
epoch = 10

for i in range(epoch):
    print(f"第{i+1}轮训练开始")
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to('mps')
        targets = targets.to('mps')
        outputs = torch.argmax(ViT(imgs), dim=-1).squeeze()
        # print(outputs, targets)
        num_same_elements = torch.sum(outputs == targets).item()
        print(num_same_elements)

