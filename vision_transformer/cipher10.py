import numpy as np
import torchvision, torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from vision_transformer import VisionTransformer
from ViT_Official import ViT1


train_data = torchvision.datasets.CIFAR10("./dataset", train=True, download=True, transform= torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform= torchvision.transforms.ToTensor())


train_dataloader = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=0, drop_last=False)
test_dataloader = DataLoader(dataset=test_data, batch_size=128, shuffle=True, num_workers=0, drop_last=False)

img_size = (32, 32)
# ViT = ViT1().to('mps')
ViT = VisionTransformer(img_size, 4, 2, 3, 10, device='mps').to(device='mps')
# ViT = torch.load("patch_16.pth", map_location=torch.device('mps'))

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(ViT.parameters(), lr=1e-2)

writer = SummaryWriter('logs_train')


# 开始训练
total_train_step = 0
total_test_step = 0
epoch = 100

for i in range(epoch):
    print(f"第{i+1}轮训练开始")
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to('mps')
        targets = targets.to('mps')
        targets = nn.functional.one_hot(targets, 10).float()
        outputs = ViT(imgs).squeeze()
        # print(outputs.shape, targets.shape)
        # print(f"outputs: {outputs},\n targets: {targets}")
        # print(f"outputs shape: {outputs.shape},\ntargets shape: {targets.shape}")
        loss = loss_function(outputs, targets)
        # print(f'loss: {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 10 == 0:
            print(f"训练次数: {total_train_step}, loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    with torch.no_grad():
        total_test_loss = 0
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to('mps')
            targets = targets.to('mps')
            targets = nn.functional.one_hot(targets, 10).float()
            outputs = ViT(imgs).squeeze()
            loss = loss_function(outputs, targets)
            total_test_loss += loss
        print(f'total test loss: {total_test_loss}')
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        total_test_step += 1
        print('模型已保存')
        torch.save(ViT, f"ViT_{i}.pth")

