import numpy as np

# 假设您的输入图像数组为image_array，形状为(1, 32, 32)，且第一个通道代表类别标签
image_array = np.random.randint(0, 3, (1, 32, 32))  # 随机生成示例数据

# 创建一个全零的(3, 32, 32)数组来存储独热编码
onehot_array = np.zeros((3, 32, 32))

# 将第一个通道的类别标签转换为独热编码
for i in range(3):
    onehot_array[i] = (image_array == i).astype(int)

print("原始图像数组:")
print(image_array.shape)

print("\n独热编码数组:")
print(onehot_array.shape)