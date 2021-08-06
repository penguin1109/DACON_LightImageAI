import PIL.Image as pil_image
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import PIL.Image as pil_image
import zipfile

def predict_means(img_dirs, model):
    #crop_rate = 4 # 2의 제곱수, 2, 4, 8
    #last = crop_rate - 2
    scale_h, scale_w = 153, 204
    results = []
    for i in range(len(img_dirs)):
        # position = []
        img = pil_image.open(img_dirs[i]).convert('RGB')
        #print(img.size)
        w,h = img.size[0], img.size[1] # (h, w)
        plt.imshow(img)
        plt.show()

        #stride_h = h//(4*crop_rate)
        #stride_w = w//(4*crop_rate)
        stride_h = (2448-612)//scale_h
        stride_w = (3264-816)//scale_w

        start_x, start_y = 0,0
        result_img = np.zeros((h, w, 3))
        voting_mask = np.zeros((h, w, 3))

        # 마지막 이미지 제외하고 iteration
        for i in range(13): # height, y
            for j in range(13): # width, x
                end_x, end_y = start_x + 816, start_y + 612
                # position.append([start_x, start_y])
                crop_img = img.crop((start_x, start_y, end_x, end_y))
                crop_img = np.array(crop_img).astype(np.float32)
                crop_img = crop_img/255.0
                crop_img = transforms.ToTensor()(crop_img)

                pred = model(torch.unsqueeze(crop_img, 0).to(device))*255
                pred = pred[0, :, :, :].to('cpu').detach().permute(1, 2, 0).numpy()

                #print(pred.shape)
                #plt.imshow(pred)
                #plt.show()
                #result_img[start_y:start_y + 612, start_x:start_x + 816, :] += pred[:, :]
                #voting_mask[start_y:start_y + 612, start_x:start_x + 816, :] += 1
                

                for c in range(3):
                    result_img[start_y:start_y + 612, start_x:start_x + 816, c] += pred[:, :, c]
                    voting_mask[start_y:start_y + 612, start_x:start_x + 816, c] += 1
                start_x += scale_w
            start_y += scale_h
            start_x = 0

        result_img = result_img/voting_mask
        #print(result_img)
        result_img = result_img.astype(np.uint8)
        plt.imshow(result_img)
        plt.show()

        results.append(result_img)
    return results

def make_mean_submission(result):
    import zipfile
    os.makedirs('submission', exist_ok = True)
    sub_imgs = []
    for i, img in enumerate(result):
        img = (img).astype(np.uint8)
        path = f'test_{20000+i}.png'
        #cv2.imwrite(path, img)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        sub_imgs.append(path)
    submission = zipfile.ZipFile('submission.zip', 'w')
    for path in sub_imgs:
        submission.write(path)
    submission.close()







