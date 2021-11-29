import os
import cv2
import numpy as np


def batch_crop(list_file, save_dir):
    """
    通过bbox切割图片，每张图片可能有多张人脸。
    list_file中每个item的格式如下：
    image_path  bbox1  bbox2 ...
    """
    with open(list_file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:

            # 先统一读取图片
            image_path = line.strip().split(" ")[0]
            print(image_path)
            img = cv2.imread(image_path)
            if img is None:
                print('None type img')
                return None

            # 保存所有的bbox信息
            parts = line.strip().split(" ")[1:]
            if not len(parts) % 4 == 0:
                print("bbox count error: ", image_path) 
            bboxes = []
            for i in range(int(len(parts) / 4)):
                bboxes.append(parts[i*4: i*4+4]) 

            # 对每个bbox分别切图
            for i,bbox in enumerate(bboxes):
                new_img = crop(img, bbox)

                # 如果大小超过1024了，就缩放为一半
                h,w = new_img.shape[:2]
                while max(h, w) > 1024:
                    new_img = cv2.resize(new_img, (int(w/2), int(h/2)))
                    h,w = new_img.shape[:2]

                # 保存裁剪后的图片
                save_name = os.path.basename(image_path).split(".")[0] + "_" + str(i+1) + ".png"
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path, new_img)
    return


def crop(img, bbox, scale=3.5):
    """
    x_min, y_min, face_x, face_y = bbox
    """
    shape = img.shape
    h, w = shape[:2]

    bbox = [int(x) for x in bbox]
    x_min, y_min, face_x, face_y = bbox
    x_mid = x_min + face_x / 2.0
    y_mid = y_min + face_y / 2.0 - face_y / 6.0       # 中心点在y轴方向网上挪了一些

    if x_mid<=0 or y_mid<=0 or face_x<=0 or face_y<=0:
        return img

    x_min = int( max(x_mid - face_x * scale / 2.0, 0) )
    x_max = int( min(x_mid + face_x * scale / 2.0, w) )
    y_min = int( max(y_mid - face_y * scale / 2.0, 0) )
    y_max = int( min(y_mid + face_y * scale / 2.0, h) )
    if x_min>=x_max or y_min>=y_max:
        return img

    return img[y_min:y_max, x_min:x_max, :]



if __name__ == "__main__":

    list_file = "/home/data/data/apple_project/list/repeat_20211018_bbox.txt"
    save_dir = "/home/data/data/apple_project/images_cropped/repeat_20211018"
    batch_crop(list_file, save_dir)
