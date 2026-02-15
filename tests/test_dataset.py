import sys
import os
# --- 新增代码开始 ---
# 获取当前文件的目录 (models/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录 (Semcom_Infra/)
parent_dir = os.path.dirname(current_dir)
# 将父目录加入到系统路径中
sys.path.append(parent_dir)
# --- 新增代码结束 ---

import torch
from core.dataset import *




if __name__ == "__main__":
    # test get_standard_transforms
    dataset_transform = get_standard_transforms(crop_size=64, is_train=True)

    # test ImageDataset
    train_dataset = ImageDataset(source="F:/科研/Semcom_Infra/data/UIEB/train",
                                 source_type= "local",
                                 transform=dataset_transform)
    
    train_loader = get_dataloader(dataset=train_dataset,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=4)
    
    batch = next(iter(train_loader))
    print(batch.size())