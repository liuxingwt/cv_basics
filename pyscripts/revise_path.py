import os
import numpy as np


def main(src_list, dst_list, new_prefix="/home/data4/"):
    """
    add prefix to each item
    """
    with open(src_list, 'r') as fr, open(dst_list, 'w') as fw:
        lines = fr.readlines()
        for i,line in enumerate(lines):
            new_line = os.path.join(new_prefix, line)
            # new_line = line.split('/home/data/raid0/')[1]
            fw.writelines(new_line)
            if (i % 1000 == 0):
                print("count:  ", i)
    return

  

if __name__ == "__main__":
  
    src_list = "/home/projects/list/list_72/RNB_nir_train_rbg.txt"
    dst_list = "/home/projects/list/list_72/RNB_nir_train_rbg_v2.txt"
    main(src_list, dst_list)

    
