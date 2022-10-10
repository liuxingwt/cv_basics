# coding=utf-8
import cv2
import requests
import base64
import numpy as np
from glob import glob



def get_info(img_path, url="https://dllab.m*******.com/demo/face/api/liveness/"):
    with open(img_path, 'rb') as fr:
        img_base64 = base64.b64encode(fr.read())
        img_base64 = str(img_base64, 'utf-8')
        img_base64 = "data:image/webp;base64," + img_base64
    query_dict = {"image_base64": img_base64}
    res = requests.post(url, data=query_dict)
    return res.json()



def batch_process(src_dir, dst_file):
    img_paths = glob(src_dir)
    # print(img_paths)
    res = []

    for img_path in img_paths:
        info = get_info(img_path) 
        print(info)
        # exit(0)
        try:
            res.append(img_path + "   " + str(info["livenessScore"]))
        except KeyError:
            continue

    with open(dst_file, 'w') as fw:
        for item in res:
            print(item)
            fw.writelines(item + "\n")
    return



if __name__ == "__main__":

    src_dir = "/home/data/data/apple_project/images_cropped/repeat_20211018/*.png"
    dst_file = "/home/data/data/apple_project/list/repeat_20211018_bbox_result.txt"
    batch_process(src_dir, dst_file)
