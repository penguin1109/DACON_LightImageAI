import cv2
import matplotlib.pyplot as plt

BOX_COLOR = (255, 0, 0) # red

def visualize_bbox(img, bbox, color = BOX_COLOR, thickness = 2):
    """visualizes a bounding box on the lights"""
    x_min, y_min, w, h = bbox
    x_min, y_min, x_max, y_max = int(x_min) , int(y_min), int(x_min + w), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness = thickness)
    
    return img

def visualize(img, bboxs):
    """"shows boxed img on screen"""
    img = img.copy()
    for bbox in bboxs:
        visualize_bbox(img, bbox)
    plt.figure(figsize = (10, 10))
    plt.axis('off')
    plt.imshow(img)


import cv2, torch
import numpy as np
from matplotlib import pyplot as plt

class ClaheTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img):
        # 흑백 이미지로 바꾸어야 한다.
        img = cv2.imread(filename, 0);
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
        img_2 = clahe.apply(img)
        return img_2

