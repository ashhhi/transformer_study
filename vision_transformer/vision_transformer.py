from encoder import Encoder, PositionwiseFeedForward
from embedding import ImageEmbedding
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from layernorm import LayerNorm
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, n_head, n_layer, n_class, device):
        super(VisionTransformer, self).__init__()
        d_model = patch_size * patch_size * 3
        self.embedding = ImageEmbedding(img_size, patch_size, device)
        self.drop1 = nn.Dropout(0.2)
        self.encoder = Encoder(patch_size, n_head, n_layer)
        self.norm = LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, n_class)

    def MLP_head(self, x):
        x = self.fc1(x)
        x = F.softmax(x, dim=-1)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = self.drop1(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x[:, :1, :]
        x = self.MLP_head(x)
        return x


if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # This always results in MPS

    image = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Ice Plant(different resolution)/320*240/2561716965140_.pic.jpg'
    image = Image.open(image)
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    print('输入图像尺寸：', image.shape)

    trans = VisionTransformer(patch_size=16, n_head=8, n_layer=8, n_class=3, device=device).to(device=device)
    x = trans(image)
    print('经过transformer之后的尺寸：', x.shape)

    # embedding = ImageEmbedding(16, device='mps')
    # x = embedding(image)
    # print(x.shape)
