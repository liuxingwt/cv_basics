import os
import cv2
import numpy as np
import glob

def detection(src_file, dst_file, dst_info_path):
    # 读取原始的灰度图片
    gray = cv2.imread(src_file, -1)
    edges = cv2.Canny(gray,50,150,apertureSize=3)    #apertureSize是sobel算子大小，只能为1,3,5，7
    lines = cv2.HoughLines(edges,1,np.pi/180,200)  #函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    # HoughLinesP概率霍夫变换（是加强版）使用简单，效果更好，检测图像中分段的直线（而不是贯穿整个图像的直线）
    # lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50,maxLineGap=10)  #函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    print("line numbers: {0:d}".format(len(lines)))

    new_lines = []
    for line in lines:
        rho,theta = line[0]    #获取极值ρ长度和θ角度
        a = np.cos(theta)    #获取角度cos值
        b = np.sin(theta)    #获取角度sin值
        x0 = a * rho    #获取x轴值
        y0 = b * rho    #获取y轴值　　x0和y0是直线的中点
        x1 = int(x0 + 1000*(-b))    #获取这条直线最大值点x1
        y1 = int(y0 + 1000*(a))    #获取这条直线最大值点y1
        x2 = int(x0 - 1000 * (-b))    #获取这条直线最小值点x2　　
        y2 = int(y0 - 1000 * (a))    #获取这条直线最小值点y2　　其中*1000是内部规则

        print("rho: {0:.3f}, theta: {1:.3f}".format(rho,theta), x1, y1, x2, y2)
        new_lines.append([theta, x1, y1, x2, y2])
        cv2.line(gray,(x1,y1),(x2,y2),(0,0,255),2)    #开始画直线
    
    # 保存画完线的图片
    cv2.imwrite(dst_file, gray)
    # 保存直线信息
    with open(dst_info_path, 'w') as fw:
        for line in new_lines:
            fw.writelines(" ".join( [str(i) for i in line]) + "\n")
    return
    

def batch_detection(src_dir, dst_dir, dst_info):
    src_files = glob.glob




if __name__ == "__main__":

    # 原始图片路径
    src_file =  "/home/data/data/ly_zhijia/data/20210316_testset_3000/ng/01_01_211.bmp"
    # 画完直线的图片的保存路径
    dst_dir = "/home/data/data/ly_zhijia/data/20210316_testset_3000_processed_2/ng"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    dst_file = os.path.join(dst_dir, os.path.basename(src_file))
    # 存储该图片所有直线信息
    dst_info = "/home/data/data/ly_zhijia/data/20210316_testset_3000_processed_3/ng"
    if not os.path.exists(dst_info):
        os.makedirs(dst_info)
    dst_info_path = os.path.join(dst_info, os.path.basename(src_file).split(".")[0] + ".txt")

    detection(src_file, dst_file, dst_info_path)
