import torch
import torch.nn.functional as F

# 假设 mask 是形状为 (224, 224) 的 torch.tensor，类别数量为 3
mask = torch.randn(224 ,224)
print(mask)# 示例随机生成一个 (224, 224) 的 mask

