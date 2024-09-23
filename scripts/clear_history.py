# clear fundus result 
import shutil


for i in range(1, 5):
    result_dir = f"/storage/zhipengdeng/data/segmentation/fundus_dofe/fed_semi/client_{i}/data/img/result"
    shutil.rmtree(result_dir)
    print(f"remove {result_dir}")
