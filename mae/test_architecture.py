import models_mae, models_vit


# mae = models_mae.__dict__['mae_vit_large_patch16'](norm_pix_loss=False)
# print(mae)
#
# vit = models_vit.__dict__['vit_large_patch16'](
#         num_classes=1000,
#         global_pool=False,
#     )
# print(vit)

from torchvision.datasets import ImageFolder

dataset = ImageFolder('/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/MAE_test/train')
print(dataset.class_to_idx)