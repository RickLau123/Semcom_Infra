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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import torch
from torch.utils.data import DataLoader, Subset

from core.codec import CNNDecoder, CNNEncoder
from core.modem import AnalogModem
from core.channel import AWGNChannel
from core.dataset import ImageDataset, get_standard_transforms, get_dataloader
from core.system import SemComSystem
from core.evaluator import SemComEvaluator

# data
uieb_test_dir = os.path.join(parent_dir, "data", "UIEB", "train")

test_transform = get_standard_transforms(crop_size=128, is_train=False)

test_dataset = ImageDataset(source=uieb_test_dir,
                            source_type='local',
                            transform=test_transform)

batch_size = 8
num_workers = 0

test_loader = get_dataloader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

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

# Testing
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Semcom_evaluator = SemComEvaluator(system=JSCC,
                                   test_loader=test_loader,
                                   device=device)

Semcom_evaluator.load_weights('F:/科研/Semcom_Infra/checkpoints/test_training/best.pth')

results = Semcom_evaluator.run(
        snr_list=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        save_dir='results/deepjscc',
    )
Semcom_evaluator.plot_results(
        snr_list=list(results.keys()),
        psnr_list=list(results.values()),
        save_dir='results/deepjscc',
    )