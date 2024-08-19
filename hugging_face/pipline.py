from datasets import load_dataset
from transformers import pipeline

vision_classifier = pipeline(model="google/vit-base-patch16-224")
preds = vision_classifier('/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Smart-Farm/image/0616_png_jpg.rf.c3fc359315fdb2728b07532193931900.jpg')
print(preds)

