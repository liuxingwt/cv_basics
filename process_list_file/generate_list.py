import os 

def generate_list(src_dir, dst_path):
    """
    把src_dir下的所有文件遍历一次，将图片的路径保存在dst_path中
    """
    dst_dict = ["bmp", "jpg", "jpeg", "JPG", "JPEG", "png", "PNG"]
    res = []                                            
    for root, dirs, files in os.walk(src_dir):
        for name in files:
            file_path = os.path.join(root, name)                                    
            if file_path.split(".")[-1] in dst_dict:
                res.append(file_path)  
                                                 
    with open(dst_path, 'w') as fw:           
        for file_path in res:
            if ".ipynb_checkpoints" in file_path:
                continue
            else:
                fw.writelines(file_path + "\n")                                    
    print("{} images found in ".format(len(res)), src_dir)
    
    return                                            

                                                
                                                
if __name__ == '__main__':
    dir_path = "/home/data/data/wkl/images1024x1024/"
    dst_file = "/home/data/data/wkl/list/ffhq.txt"
    generate_list(dir_path, dst_file)
