# Source of the following functions: 
# https://colab.research.google.com/github/orpatashnik/StyleCLIP/blob/main/notebooks/StyleCLIP_global.ipynb
# wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
# 人脸检测器：https://github.com/davisking/dlib-models/blob/master/mmod_human_face_detector.dat.bz2

import os
import numpy as np
import scipy
import scipy.ndimage
import dlib
import cv2
import PIL
import PIL.Image


def batch_run(src_list, dst_dir, sep):
    """
    把src_list中的图片，经过对齐和裁剪后，保存到dst_dir中
    """
    with open(src_list, 'r') as fr:
        lines = fr.readlines()
        for i,line in enumerate(lines):
            image_path = line.strip().split(" ")[0]
            new_path = os.path.join(dst_dir, image_path.split(sep)[1])
            
            # 已经处理过的图片
            if os.path.exists(new_path):
                continue
            # 未处理但对应目录不存在时
            new_dir = os.path.dirname(new_path)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            # 单张图片执行
            run(image_path, new_path)
            if i % 100 == 0:
                print("current count {} / {}".format(i, len(lines)))
    return



def run(image_path, saved_path):
    """
    使用dlib 检测人脸框，并进行裁剪
    """
    # 用dlib检测人脸框
    detector = dlib.get_frontal_face_detector()
    img = dlib.load_rgb_image(image_path)
    rects = detector(img, 1)
    
    # 重新用cv2处理
    img = cv2.imread(image_path)

    bboxes = []
    # 对于不存在人脸的情况
    if len(rects) == 0:     
        h, w, c = img.shape
        bbox = [w*0.45, h*0.45, w*0.55, h*0.55,]
        bbox = [int(i) for i in bbox]
        print("no face bbox: ", bbox, image_path, h, w)
        bboxes.append(bbox)
    # 对于存在人脸的情况：首先检测出所有的人脸框并保存
    else:     
        dict_area = {}
        for (i, rect) in enumerate(rects):
            x1 = rect.left()
            y1 = rect.top()
            x2 = rect.right()
            y2 = rect.bottom()

            # 使用一个字典，保存面积：位置信息
            area = int((x2 - x1) * (y2 - y1)) 
            if not area in dict_area:
                dict_area[area] = [x1, y1, x2, y2]
            else:
                area -= 1
                dict_area[area] = [x1, y1, x2, y2]

        # 对所有人脸依据大小进行排序
        area_sort = sorted(dict_area.keys(), reverse=True)
        # print(image_path, area_sort)
        bboxes.append(dict_area[area_sort[0]])
        if len(area_sort) >= 2 and area_sort[1] >= 10000:
            bboxes.append(dict_area[area_sort[1]])

    # 裁剪并保存所有的人脸
    for i,bbox in enumerate(bboxes):
        new_path = saved_path.split(".")[0] + "_{}.".format(str(i+1)) + saved_path.split(".")[1]
        crop_single(img, bbox, new_path)
    return 



def crop_single(img, bbox, saved_path, scale=2.0, resize_dims=(224, 224)):
    """
    依据人脸框大小进行裁剪和保存
    """
    shape = img.shape
    h, w = shape[:2]

    x_min, y_min, x_max, y_max = bbox
    face_x = x_max - x_min
    face_y = y_max - y_min
    x_mid = x_min + face_x / 2.0
    y_mid = y_min + face_y / 2.0

    x_min = int( max(x_mid - face_x * scale / 2.0, 0) )
    x_max = int( min(x_mid + face_x * scale / 2.0, w) )
    y_min = int( max(y_mid - face_y * scale / 2.0, 0) )
    y_max = int( min(y_mid + face_y * scale / 2.0, h) )

    new_img = img[y_min:y_max, x_min:x_max, :]
    new_img = cv2.resize(new_img, resize_dims)
    cv2.imwrite(saved_path, new_img)
    return


    

if __name__ == "__main__":    

    # test code
    src_list = "/home/data/data/apple_project/list_train/replay.txt"
    dst_dir = "/home/data/data/apple_project/images_train/apple_replay_crop/"
    sep = "apple_replay/"
    batch_run(src_list, dst_dir, sep)
