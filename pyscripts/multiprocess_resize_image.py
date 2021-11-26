import os
import numpy as np
import cv2
from multiprocessing import Process
from multiprocessing import Array


def func(i, new_size=(512, 512)):
    """
    把原始图片resize
    对于进程i，处理第 num_process * n + i个；
    """
    for num in range(num_path):
        if num % num_process == i:
            src_path = paths[num]
            dst_path = os.path.join(dst_dir, src_path.split("/")[-1])

            img = cv2.imread(src_path)
            img = cv2.resize(img, new_size)

            # if not os.path.exists(os.path.dirname(dst_path)):
            #     os.makedirs(os.path.dirname(dst_path))
            cv2.imwrite(dst_path, img)
            print("Process {}, deal: ".format(str(i)), src_path)
    return



###################  基本参数配置  #######################################
src_list = "/home/data/data/wkl/list/ffhq.txt"
dst_dir = "/home/data/data/wkl/image512/"
############ 读取所有的path，生成一个全局的list #############################
paths = []
with open(src_list, 'r') as fr:
    lines = fr.readlines()
    for i,line in enumerate(lines):
        line = line.strip().split(" ")[0]
        paths.append(line)
num_path = len(paths)
#################### 多进程处理 ###########################################
num_process = 10
for i in range(num_process):
    p = Process(target=func, args=(i,))
    p.start()
#########################################################################

