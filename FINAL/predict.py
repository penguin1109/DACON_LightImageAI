import PIL.Image as pil_image
import numpy as np
import zipfile
import matplotlib.pyplot as plt

def predict(img_dirs, model):
    results = []
    for i in range(len(img_dirs)):
        img = pil_image.open(img_dirs[i]).convert('RGB')
        print(img.size)
        w,h = img.size[0], img.size[1] # (h, w)
        plt.imshow(img)
        plt.show()
        scale = h//612
        start_x, start_y = 0,0
        result_img = np.zeros((h, w, 3))
        print(result_img.shape)
        channel = 0
        for j in range(1, scale+1):
            for k in range(1, scale+1):
                end_x, end_y = start_x + 816, start_y + 612
                crop_img = img.crop((start_x, start_y, end_x, end_y))
                crop_img = np.array(crop_img).astype(np.float32)
                crop_img = transforms.ToTensor()(crop_img)
                pred = model(torch.unsqueeze(crop_img, 0).to(device))
                pred = pred[0, :, :, :].to('cpu').detach().permute(1, 2, 0).numpy()
                #plt.imshow(pred)
                #plt.show()
                print(pred.shape)
                for c in range(3):
                    result_img[start_y:start_y + 612, start_x:start_x + 816, c] = pred[:, :, c]
                start_x += 816
            start_y += 612
            start_x = 0
        plt.imshow(result_img)
        plt.show()
        results.append(result_img)

    return results

def make_submission(result):
    os.makedirs('submission', exist_ok = True)
    sub_imgs = []
    for i, img in enumerate(result):
        img = (img * 255).astype(np.uint8)
        path = f'test_{20000+i}.png'
        cv2.imwrite(path, img)
        sub_imgs.append(path)
    submission = zipfile.ZipFile('submission.zip', 'w')
    for path in sub_imgs:
        submission.write(path)
    submission.close()









