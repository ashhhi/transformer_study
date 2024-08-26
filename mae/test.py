import torch
import torch.nn.functional as F

# 假设 mask 是形状为 (224, 224) 的 torch.tensor，类别数量为 3
mask = torch.randint(0, 3, (224, 224))  # 示例随机生成一个 (224, 224) 的 mask

# 将类别索引转换为独热编码
one_hot_mask = F.one_hot(mask, num_classes=3).permute(2, 0, 1)

# 最终得到的 one_hot_mask 的形状为 (3, 224, 224)
print(mask.shape)