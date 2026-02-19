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
from torch.utils.data import DataLoader, Subset

from core.codec import CNNDecoder, CNNEncoder
from core.modem import AnalogModem
from core.channel import AWGNChannel
from core.dataset import ImageDataset, get_standard_transforms, get_dataloader
from core.system import SemComSystem
from core.trainer import SemComTrainer

# data: split UIEB/train into 90% train and 10% val
uieb_train_dir = os.path.join(parent_dir, "data", "UIEB", "train")

train_transform = get_standard_transforms(crop_size=128, is_train=True)
val_transform = get_standard_transforms(crop_size=128, is_train=False)

full_dataset_train = ImageDataset(
    source=uieb_train_dir,
    source_type="local",
    transform=train_transform,
)
full_dataset_val = ImageDataset(
    source=uieb_train_dir,
    source_type="local",
    transform=val_transform,
)

total_size = len(full_dataset_train)
train_size = int(total_size * 0.9)
val_size = total_size - train_size

g = torch.Generator().manual_seed(42)
all_indices = torch.randperm(total_size, generator=g).tolist()
train_indices = all_indices[:train_size]
val_indices = all_indices[train_size:]

train_dataset = Subset(full_dataset_train, train_indices)
val_dataset = Subset(full_dataset_val, val_indices)

batch_size = 8
num_workers = 0

train_loader = get_dataloader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)
val_loader = get_dataloader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)

# Semcom System
encoder_test = CNNEncoder(hidden_dims=[16, 32, 32],
                          out_symbol_channels= 32,
                          kernel_size=[5, 5, 5, 5],
                          stride=[2, 2, 1, 1])
decoder_test = CNNDecoder(hidden_dims=[32, 32, 32],
                          in_symbol_channels=32,
                          kernel_size=[5, 5, 5, 5],
                          stride=[1, 1, 2, 2]
                          )
modem_test = AnalogModem()
channel = AWGNChannel()
JSCC = SemComSystem(encoder=encoder_test,
                    decoder=decoder_test,
                    modem=modem_test,
                    channel=channel)

# training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(JSCC.parameters(), lr=1e-4)

Semcom_trianer = SemComTrainer(
    system=JSCC,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=None,
    device=device,
    config={
        "train_snr_list": [1, 4],
        "test_snr_list": [1, 4, 7],
        "log_dir": "runs/test_training",
        "checkpoint_dir": "checkpoints/test_training",
    },
)

Semcom_trianer.fit(epochs=5)

