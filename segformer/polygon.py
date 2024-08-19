import os

import numpy as np
import yaml
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET


image_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Roboflow/v6/image'
label_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Roboflow/v6/mask_segformer'

polygons_split = {}
id2label = {}
with open('id2label.txt', 'r') as f:
    data = f.readlines()
    for line in data:
        id = int(line.split()[0])
        label = line.split()[1]
        polygons_split[label]=[]
        id2label[id]=label
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
print(num_labels)


def polygon_to_mask(label_path, save_name):
    # Parse the XML label
    tree = ET.parse(label_path)
    root = tree.getroot()

    # Process each object element
    for object_element in root.findall('object'):
        # Extract information from the XML elements
        name = object_element.find('name').text
        xmin = int(object_element.find('bndbox/xmin').text)
        xmax = int(object_element.find('bndbox/xmax').text)
        ymin = int(object_element.find('bndbox/ymin').text)
        ymax = int(object_element.find('bndbox/ymax').text)

        polygon_element = object_element.find('polygon')
        polygon_points = []
        for i in range(1, 999999):
            try:
                x = float(polygon_element.find(f'x{i}').text)
                y = float(polygon_element.find(f'y{i}').text)
                polygon_points.append((x, y))
            except Exception as e:
                break

        # Print the extracted information for each object
        print('Object Name:', name)
        print('Bounding Box:', xmin, ymin, xmax, ymax)
        print('Polygon Points:', polygon_points)
        print('------------------------')

        for key in label2id.keys():
            if name == key:
                polygons_split[key].append(polygon_points)


    # Get the image size from the XML or provide it manually
    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    # Create a blank RGB mask image
    mask = Image.new('RGB', (image_width, image_height), (0, 0, 0))

    # Create a draw object
    draw = ImageDraw.Draw(mask)

    # Set colors for stem and leaf

    for key in polygons_split.keys():
        for item in polygons_split[key]:
            polygon_points = [(int(x), int(y)) for x, y in item]
            draw.polygon(polygon_points, outline=int(label2id[key]), fill=int(label2id[key]))

    # Convert the mask to a NumPy array
    mask.save(save_name)


g = os.walk(image_path)
image = []
label = []
for path, dir_list, file_list in g:
    for file_name in file_list:
        if file_name.split('.')[-1] == 'xml':
            polygon_to_mask(str(os.path.join(path, file_name)), str(os.path.join(label_path, str(file_name.replace('.xml', '.png')))))