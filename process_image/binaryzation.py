import os
import cv2
import numpy as np

def binaryzation(src_file, dst_file, low=80, high=120):

    img = cv2.imread(src_file, -1)
    # print("shape", img.shape)
    # print(img[100:110, 100:105])
    img = cv2.rectangle(img, (100,100), (110,110), (255,0,0), 2)

    img = np.where(low < img, img, 0)
    img = np.where(img < high, img, 0)

    cv2.imwrite(dst_file, img)
    return



if __name__ == "__main__":

    src_file =  "/home/data/data/ly_zhijia/data/20210316_testset_3000/ng/01_01_23.bmp"
    dst_dir = "/home/data/data/ly_zhijia/data/20210316_testset_3000_processed/ng"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    dst_file = os.path.join(dst_dir, os.path.basename(src_file))

    binaryzation(src_file, dst_file)
