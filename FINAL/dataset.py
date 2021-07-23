import torch
import torch.utils.data.dataset as Dataset
import numpy as np
import PIL.Image as pil_image
import torchvision.transforms as transforms


class CroppedDataset(torch.utils.data.Dataset):
    # read and change image size(3, 2448, 3264) -> (3, 612, 816)
    def __init__(self, x_img_dirs, y_img_dirs, split, is_train, crop_size = (612, 816)):
        # img_dirs는 이미지의 저장 경로가 담겨있는 리스트의 형태
        # crop_size는 잘라낼 이미지의 크기로, 일정하게 정해 놓을 예정
        self.input = x_img_dirs
        self.target = y_img_dirs
        self.split = int(len(self.input) * split)
        self.w, self.h = crop_size[0], crop_size[1]

        if is_train:
          self.input = self.input[:self.split]
          self.target = self.target[:self.split]
        else:
          self.input = self.input[self.split:]
          self.target = self.target[self.split:]

        self.samples = []
        # target, label이미지의 저장 경로를 하나씩 리스트에 넣어줌
        for x, y in zip(self.input, self.target):
            self.samples.append([x, y])
    
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        inputs, labels = self.samples[idx]
        in_img, label_img = pil_image.open(inputs).convert('RGB'), pil_image.open(labels).convert('RGB')
        imgtonp = lambda x: np.array(x).astype(np.float32)
        
        transform = []
        transform.append(transforms.Resize((self.w, self.h)))
        transform.append(transforms.ToTensor())
        transform = transforms.Compose(transform)

        resized_x = transform(in_img)
        resized_y = transform(label_img)

        h, w = in_img.size[0], in_img.size[1]

        scale = int(w/self.w)
        img_nums = scale ** 2

        start_x, start_y = 0, 0
        data = []
        data.append({'input' : resized_x, 'target' : resized_y})
        for i in range(1, scale+1):
            for j in range(1, scale+1):
                end_x, end_y = start_x + self.h, start_y + self.w
                crop_x = in_img.crop((start_x, start_y, end_x, end_y))
                crop_y = label_img.crop((start_x, start_y, end_x, end_y))

                crop_x = np.array(crop_x).astype(np.float32)
                crop_y = np.array(crop_y).astype(np.float32)

                # pil_image.read로 읽었기 떄문에 (h, w, 3)이므로
                # 입력으로 모델에 넣어 줄 떄에는 (3, h, w)로 바꾸어야 함

                # scale 0-255.0 -> 0-1
                crop_x = crop_x/255.0
                crop_y = crop_y/255.0

                crop_x = transforms.ToTensor()(crop_x)
                crop_y = transforms.ToTensor()(crop_y)

                data.append({'input' : crop_x, 'target' : crop_y})
                start_x += self.h
            start_y += self.w
            start_x = 0

        return data           