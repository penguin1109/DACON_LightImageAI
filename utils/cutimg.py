import os, cv2
import numpy as np
import tqdm as tqdm

# (128, 128)의 크기로 이미지 위를 slide해가면서 조각내어 주는 함수
def cut_img(img_path_list, save_path, stride, img_size):
    os.makedirs(save_path, exist_ok=True)
    num = 0
    for path in tqdm(img_path_list):
        img = cv2.imread(path)
        shape = img.shape
        for top in range(0, shape[0], stride):
            for left in range(0, shape[1], stride):
                piece = np.zeros([img_size, img_size, 3], np.unit8)
                temp = img[top:top+img_size, left:left + img_size, :]
                piece[:temp.shape[0], :temp.shape[1], :] = temp
                np.save()


# 이미지의 일부분을 cutout 함수의 적용을 통해서 지정해주는 개수만큼 검은색으로 path를 붙이는 느낌으로 생각
# Albumentation의 cutout함수를 적용해도 됨
def cutout(images, cut_length):
    """
    Perform cutout augmentation from images.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param cut_length: int, the length of cut(box).
    :return: np.ndarray, shape: (N, h, w, C).
    """

    H, W, C = images.shape[1:4]
    augmented_images = []
    for image in images:    # image.shape: (H, W, C)
        image_mean = image.mean(keepdims=True)
        image -= image_mean

        mask = np.ones((H, W, C), np.float32)

        y = np.random.randint(H)
        x = np.random.randint(W)
        length = cut_length

        y1 = np.clip(y - (length // 2), 0, H)
        y2 = np.clip(y + (length // 2), 0, H)
        x1 = np.clip(x - (length // 2), 0, W)
        x2 = np.clip(x + (length // 2), 0, W)

        mask[y1: y2, x1: x2] = 0.
        image = image * mask

        image += image_mean
        augmented_images.append(image)

    return np.stack(augmented_images)    # shape: (N, h, w, C)