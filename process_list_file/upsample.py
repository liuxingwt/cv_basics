import os
import glob
from tqdm import tqdm

is_sta_label = 1
data_list = [
    # 正样本，数目10699
    ['/home/data/data/ly_zhijia/code/list/dataset2_nolarge_target_list_pos_prop01.txt', 1],
    # 负样本，数目3274
    ['/home/data/data/ly_zhijia/code/list/dataset2_nolarge_target_list_neg.txt', 3],
]

path_save = '/home/data/data/ly_zhijia/code/list/train_list/train_20210817.txt'


def process():
    lines_all = []
    for txt_name_full, repeat_num in data_list:
        with open(txt_name_full, 'r') as fid:
            lines = fid.readlines()
        lines = lines * repeat_num
        print(txt_name_full, len(lines))
        lines_all += lines
    
    with open(path_save, 'w') as fid:
        fid.writelines(lines_all)

    if not is_sta_label:
        return
    
    pos_num = 0
    neg_num = 0
    
    for line in tqdm(lines_all):
        label_str = line.split()[1]
        if label_str == '1':
            pos_num += 1
        if label_str == '0':
            neg_num += 1
        assert(label_str == '1' or label_str == '0')
    
    print(pos_num, neg_num)


    
if __name__ == '__main__':
    process()
    
