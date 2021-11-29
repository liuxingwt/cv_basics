import os
import random
import shutil


def split_file_in_prop(src_list, list1, list2, list1_prop=0.1):
    """
    split src_list into two parts
    src_list = "../train_list_pair/train_all_20201230.txt" 
    list1 = "../train_list_pair/train_all_20201230_prop001.txt"
    list2 = "../train_list_pair/train_all_20201230_prop099.txt"
    split_file_in_prop(src_list, list1, list2)
    """
    with open(src_list, 'r') as fr, open(list1, 'w') as fw1,  open(list2, 'w') as fw2: 
        lines = fr.readlines()
        begin = 0
        end = len(lines) - 1
        need_count = int(len(lines) * list1_prop) - 1
        result_list = random.sample(range(begin, end), need_count)
        for i, line in enumerate(lines):
            if i in result_list:
                fw1.writelines(line)
            else:
                fw2.writelines(line)
    return


def pick_file_in_prop(src_list, dst_list, dst_dir, prop=0.02):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    with open(src_list, 'r') as fr, open(dst_list, 'w') as fw: 
        lines = fr.readlines()
        begin = 0
        end = len(lines) - 1
        need_count = int(len(lines) * prop) - 1
        result_list = random.sample(range(begin, end), need_count)
        for i, line in enumerate(lines):
            if i in result_list:
                src_path = line.split(" ")[0]
                img_name = str(random.random()) +  "_" + os.path.basename(src_path)
                dst_path = os.path.join(dst_dir, img_name)
                shutil.copy(src_path, dst_path)

                fw.writelines(img_name + " " + line.split(" ")[1] + "\n")
                print(i)
    return


def pick_file(src_list, dst_dir):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    with open(src_list, 'r') as fr: 
        lines = fr.readlines()
        for i, line in enumerate(lines):
            src_path = line.split(" ")[0]
            img_name = os.path.basename(src_path)
            dst_path = os.path.join(dst_dir, img_name)
            shutil.copy(src_path, dst_path)
            print(i)
    return





if __name__ == "__main__":
    # src_list = "/home/projects/list/list_mask/MLT_door_mask_0323_remake_v3.txt" 
    # list1 = "/home/projects/list/list_mask/MLT_door_mask_0323_remake_v3_test.txt" 
    # list2 = "/home/projects/list/list_mask/MLT_door_mask_0323_remake_v3_train.txt" 
    # split_file_in_prop(src_list, list1, list2)

    src_list = "/home/projects/list/list_72/cbsr_mas_v6_hifi-mask-test_bbox.txt"
    dst_list  = "/home/projects/test_cbsr_prop002.txt"
    dst_dir="/home/projects/test_cbsr_prop002"
    pick_file_in_prop(src_list, dst_list, dst_dir)

    # src_list = "/home/projects/list/list_mask/MLT_door_mask_0323_all_test_bbox.txt"
    # dst_dir = "/home/projects/test_mask/"
    # pick_file(src_list, dst_dir)
    
