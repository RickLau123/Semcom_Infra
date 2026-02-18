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

from core.codec import CNNDecoder, CNNEncoder
from core.modem import AnalogModem
from core.channel import AWGNChannel
from core.system import SemComSystem

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

x = torch.randn(4, 3, 64, 64) * 255

if __name__ == '__main__':
    JSCC = SemComSystem(encoder=encoder_test,
                        decoder=decoder_test,
                        modem=modem_test,
                        channel=channel)
    output = JSCC(x, 10)
    print('test done!')
    print(JSCC)