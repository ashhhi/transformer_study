import torch
from torch import nn
from torchvision import transforms
from PIL import Image

class ImageEmbedding(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, device='cpu'):
        super(ImageEmbedding, self).__init__()
        self.patch_size = patch_size
        self.d_model = patch_size * patch_size * 3
        n_patch = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.conv = nn.Conv2d(3, self.d_model, self.patch_size, self.patch_size, device=device)
        self.flatten = nn.Flatten(2)
        self.device = device
        self.cls_token = nn.Parameter(torch.rand(self.d_model, 1, device=self.device, dtype=torch.float))
        self.pos_token = nn.Parameter(torch.rand(1, n_patch + 1, device=self.device, dtype=torch.float))

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv(x)
        x = self.flatten(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=-1)
        x += self.pos_token
        x = torch.permute(x, (0, 2, 1))
        return x

if __name__ == '__main__':
    image = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Ice Plant(different resolution)/320*240/2561716965140_.pic.jpg'
    image = Image.open(image)
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    print(image.shape)
    embedding = ImageEmbedding(16, device='mps')
    x = embedding(image)
    print(x.shape)
