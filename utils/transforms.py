import cv2, random
import torch
import torchvision.transforms as transforms
def contrast(img):
    """returns change in contrast for the light"""
    img = cv2.imread(img)
    # 밝기의 대비를 주고자 하는데, 그렇게 하기 위해서는 beta값이 0이어야 한다
    # 그리고 alpha값은 1.2와 2사이의 랜덤한 수로 설정해 주어야 한다. 
    new_img = cv2.convertScaleAbs(img, alpha = random.random(1.2, 2), beta = 0)
    return new_img

    
def get_transform(train = True, mean = None, std = None):
    transform = []
    normalize = transforms.Normalize(
        mean, std, inplace = True
    )
    transform.append(transforms.Resize(size = (256, 256)))
    if train:
        transform.append(transforms.RandomSizedCrop(128))
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(normalize)
    else:
       transform.append(transforms.ToTensor())
       transform.append(normalize)
    
    return transforms.Compose(transform)


class ContrastTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        img = transforms.functional.adjust_contrast(img = img, contrast_factor = random.uniform(1.2, 2))
        return img

def pil_loader(path, clahe = True, img_size=256): 
  if not clahe:
    with open(path, "rb") as f:
      img = Image.open(f)
      return img.convert("RGB") # img pixel의 순서를 RGB의 순서로 바꾸어서 저장할 수 있도록 한다.
  else:
    img = cv2.imread(path)
    return img

class ClaheTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img):
        # 흑백 이미지로 바꾸어야 한다.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
        img_2 = clahe.apply(img)
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_2)