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
from torch import nn

from core.channel import AWGNChannel
from core.modem import AnalogModem
# from models.cnn_module import CNNEncoder, CNNDecoder
from core.codec import CNNEncoder, CNNDecoder
from core.utils import tensor_to_image

def test_pipeline(x: torch.Tensor):
    """
    Test the semantic communication pipeline.

    This function simulates the full pipeline of semantic communication,
    including encoding, modulation, channel transmission, demodulation,
    and decoding. It verifies the integration of all components.

    Args:
        x: Input image tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: Reconstructed image tensor of shape (B, C, H, W).

    Pipeline:
        1. Encode:    x -> x_hat (semantic symbols)
        2. Modulate:  x_hat -> x_hat_complex (power-normalized symbols)
        3. Channel:   x_hat_complex -> temp (noisy symbols)
        4. Demodulate: temp -> z (received symbols)
        5. Decode:    z -> z_hat (reconstructed image)

    Example:
        >>> x = torch.randn(4, 3, 32, 32)
        >>> z_hat = test_pipeline(x)
        >>> z_hat.shape  # (4, 3, 32, 32)
    """
    # --- Initialize components ---
    encoder_test = CNNEncoder()  # Semantic encoder
    decoder_test = CNNDecoder()  # Semantic decoder
    channel = AWGNChannel()      # AWGN channel model
    modem = AnalogModem()        # Analog modem

    # --- Step 1: Semantic Encoding ---
    # Input: (B, C, H, W) -> Output: (B, 2, k, H', W')
    x_hat = encoder_test(x)

    # --- Step 2: Modulation ---
    # Normalize power: (B, 2, k, H', W') -> (B, 2, k, H', W')
    x_hat_complex = modem(x_hat)

    # --- Step 3: Channel Transmission ---
    # Add noise: (B, 2, k, H', W') -> (B, 2, k, H', W')
    temp = channel(x_hat_complex, 10)

    # --- Step 4: Demodulation ---
    # Pass-through: (B, 2, k, H', W') -> (B, 2, k, H', W')
    z = modem.demodulate(temp)

    # --- Step 5: Semantic Decoding ---
    # Decode: (B, 2, k, H', W') -> (B, C, H, W)
    z_hat = tensor_to_image(decoder_test(z))

    print('test done!')

    return z_hat


if __name__ == "__main__":
    """
    Main entry point for testing the pipeline.

    This script generates random input images, runs them through the
    semantic communication pipeline, and prints the result.
    """
    x = torch.randn(4, 3, 32, 32)  # Random input images
    test_pipeline(x)

