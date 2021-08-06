import torch as torch
import torchvision
from torch.utils.data import Dataset
# 이런식으로 모듈 불러오는 방법도 제대로 숙지하고 있어야 한다.
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, x, y, is_train = True, split = 1):
    # is_train = True일 때와 아닐때(즉, validataion data로 이용할 예정)
    # 일단 default값은 1로 지정해 줌
        self.x_img = x
        self.y_img = y
    
        split = int(len(self.x_img) * split)
    
        if is_train:
            train = True
            self.x_img = self.x_img[:split]
            self.y_img = self.y_img[:split]
        else:
            train = False
            self.x_img = self.x_img[:split]
            self.y_img = self.y_img[:split]
    
        self.samples = []
        for x, y in zip(self.x_img, self.y_img):
            self.samples.append([x, y])

        self.transform = get_transform(train = train)

    def __len__(self):
    # 이미지의 파일 경로가 담긴 리스트를 입력할 예정
        return len(self.x_img)

    def __getitem__(self, idx):
        input, label = self.samples[idx]

        input = self.transform(pil_loader(input))
        label = self.transform(pil_loader(label))

        return [input, label]

def pil_loader(path, img_size=256): 
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB") # img pixel의 순서를 RGB의 순서로 바꾸어서 저장할 수 있도록 한다.
    except FileNotFoundError as e:
        raise FileNotFoundError(e)


# albumention을 적용하는 경우
class Dataset_1(Dataset):
    def __init__(self, x, y, x_transforms = None, common_transforms = None):
        super().__init__
        self.x_img = x
        self.y_img = y
        self.x_trans = x_transforms
        self.common_trans = common_transforms
    
    def __len__(self):
        return len(self.x_img)
    
    def __getitem__(self, idx):
        x = self.x_img[idx]
        y = self.y_img[idx]
     
        if self.x_trans != None:
            x_transformed = self.x_transforms(image = x)
            x = x_transformed['image']

        if self.common_trans != None:
            transformed  = self.common_trans(image = x, target_img = y)
            x = transformed['image']
            y = transformed['target_img']

        return x, y

import Albumentations as albu

common_transform = albu.Compose([
    albu.HorizontalFlip(p = 0.5),
    albu.VerticalFlip(p = 0.5),
    albu.Cutout(num_holes = 8, max_h_size = 50, max_h_size = 50, p = 1.)
], additional_targets = {'target_img':'image'})

x_transform = albu.Compose([
    albu.Posterize(p=1.)
])