"""
Base Modem Module for SemCom_Infra.

This module provides the abstract base class for modem operations including:
- Power normalization (ensuring E[x^2] = 1)
- Complex <-> Real conversion for differentiable physical layer processing
- OFDM modulation/demodulation (to be implemented in subclasses)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class BaseModem(nn.Module):
    """
    Abstract base class for Modem modules.
    
    The Modem is responsible for:
    1. Power normalization to satisfy E[x^2] = 1 constraint
    2. Complex <-> Real tensor conversion
    3. Physical layer modulation (e.g., OFDM) - implemented in subclasses
    
    Note:
        Complex signals are represented as real tensors with shape (B, 2, ...),
        where dim=1 contains [I, Q] (in-phase and quadrature components).
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize BaseModem.
        
        Args:
            epsilon: Small constant to prevent division by zero in normalization.
                    Default: 1e-8
        """
        super(BaseModem, self).__init__()
        self.epsilon = epsilon
    
    def power_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input signal to satisfy power constraint E[x^2] = 1.
        
        This method computes the average power per sample in the batch and
        normalizes each sample independently to unit power.
        
        Formula:
            x_norm = x / sqrt(E[x^2] + epsilon)
        
        Args:
            x: Input signal tensor of shape (B, ...).
               B is batch size, ... represents arbitrary dimensions.
        
        Returns:
            Normalized tensor of the same shape as input, where each sample
            satisfies E[x^2] ≈ 1.
        
        Shape:
            - Input: (B, *) where * means any number of dimensions
            - Output: (B, *) same shape as input
        
        Example:
            >>> modem = BaseModem()
            >>> x = torch.randn(4, 2, 16, 16)  # (B=4, C=2, H=16, W=16)
            >>> x_norm = modem.power_normalize(x)
            >>> # Verify: torch.mean(x_norm[0]**2) ≈ 1.0
        """
        batch_size = x.shape[0]
        
        # Compute average power for each sample: E[x^2]
        # Shape: (B, ...) -> (B,)
        power = torch.mean(
            x.view(batch_size, -1) ** 2,  # (B, N) where N = product of other dims
            dim=1,  # Average over all elements in each sample
            keepdim=True  # (B, 1) for broadcasting
        )
        
        # Normalize: x / sqrt(power + epsilon)
        # Shape: (B, 1) -> (B, 1, 1, ...) for broadcasting
        power = power.view(batch_size, *([1] * (x.ndim - 1)))  # Expand to match x dims
        x_normalized = x / torch.sqrt(power + self.epsilon)
        
        return x_normalized
    
    @staticmethod
    def complex_to_real(x: torch.Tensor) -> torch.Tensor:
        """
        Convert complex-valued tensor to real-valued representation.
        
        PyTorch complex tensors (dtype=torch.complex64/128) are converted to
        real tensors by stacking real and imaginary parts along a new dimension.
        
        Args:
            x: Complex tensor of shape (B, C, H, W) or (B, C, ...).
               Must have dtype torch.complex64 or torch.complex128.
        
        Returns:
            Real tensor with shape (B, 2, C, H, W) where:
                - dim=1, index 0 → Real part (I)
                - dim=1, index 1 → Imaginary part (Q)
        
        Shape:
            - Input: (B, C, H, W) [complex]
            - Output: (B, 2, C, H, W) [real]
        
        Example:
            >>> x_complex = torch.randn(4, 3, 32, 32, dtype=torch.complex64)
            >>> x_real = BaseModem.complex_to_real(x_complex)
            >>> x_real.shape  # (4, 2, 3, 32, 32)
            >>> # x_real[:, 0, ...] is real part, x_real[:, 1, ...] is imaginary part
        
        Note:
            This convention (B, 2, C, H, W) ensures compatibility with
            differentiable channel models that expect real-valued inputs.
        """
        assert torch.is_complex(x), "Input must be a complex tensor"
        
        # Stack real and imaginary parts along dim=1
        # Shape: (B, C, H, W) -> (B, 2, C, H, W)
        x_real = torch.stack([x.real, x.imag], dim=1)
        
        return x_real
    
    @staticmethod
    def real_to_complex(x: torch.Tensor) -> torch.Tensor:
        """
        Convert real-valued representation back to complex tensor.
        
        This is the inverse operation of complex_to_real.
        
        Args:
            x: Real tensor of shape (B, 2, C, H, W) where:
                - dim=1, index 0 → Real part (I)
                - dim=1, index 1 → Imaginary part (Q)
        
        Returns:
            Complex tensor with shape (B, C, H, W) and dtype torch.complex64.
        
        Shape:
            - Input: (B, 2, C, H, W) [real]
            - Output: (B, C, H, W) [complex]
        
        Example:
            >>> x_real = torch.randn(4, 2, 3, 32, 32)
            >>> x_complex = BaseModem.real_to_complex(x_real)
            >>> x_complex.shape  # (4, 3, 32, 32)
            >>> x_complex.dtype  # torch.complex64
        
        Raises:
            AssertionError: If input doesn't have dim=1 with size 2.
        """
        assert x.shape[1] == 2, (
            f"Expected dim=1 to have size 2 [real, imag], got {x.shape[1]}"
        )
        
        # Extract real and imaginary parts
        # Shape: (B, 2, C, H, W) -> 2 × (B, C, H, W)
        real_part = x[:, 0, ...]  # (B, C, H, W)
        imag_part = x[:, 1, ...]  # (B, C, H, W)
        
        # Construct complex tensor
        # Shape: (B, C, H, W) [complex]
        x_complex = torch.complex(real_part, imag_part)
        
        return x_complex
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (to be implemented in subclasses).
        
        Subclasses should implement specific modulation schemes (e.g., OFDM).
        
        Args:
            x: Input tensor from encoder.
        
        Returns:
            Modulated signal ready for channel.
        
        Raises:
            NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("Subclasses must implement forward method")


class AnalogModem(BaseModem):
    """
    Analog Modem for Semantic Communication.
    
    This modem implements the physical layer interface for analog JSCC
    (Joint Source-Channel Coding). Unlike digital modems, it does not
    perform quantization or discrete modulation. Instead, it relies on
    the neural network codec to directly operate on continuous-valued symbols.
    
    Key Features:
        - Transmission (forward): Power normalization to satisfy E[x^2] = 1.
        - Reception (demodulate): Identity/pass-through operation.
    
    Rationale for Pass-through Demodulation:
        In analog JSCC, the decoder network is trained end-to-end to be
        robust against channel noise and amplitude variations. Therefore,
        manual de-normalization or equalization is unnecessary and may
        even degrade performance. The decoder learns to adapt to the
        received signal distribution implicitly.
    
    Note:
        For digital JSCC or OFDM-based schemes, the demodulate() method
        should be overridden to perform FFT, LLR calculation, or other
        digital demodulation steps.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize AnalogModem.
        
        Args:
            epsilon: Small constant to prevent division by zero in normalization.
                    Default: 1e-8
        """
        super(AnalogModem, self).__init__(epsilon=epsilon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Modulate symbols for transmission (Tx side).
        
        This method performs power normalization to ensure the transmitted
        signal satisfies the power constraint E[x^2] = 1, which is required
        for fair SNR comparison across different models and configurations.
        
        Args:
            x: Input symbol tensor of shape (B, 2, k, H', W') from encoder.
               These are un-normalized semantic symbols.
        
        Returns:
            Power-normalized signal of the same shape, ready for channel.
        
        Shape:
            - Input: (B, 2, k, H', W')
            - Output: (B, 2, k, H', W')
        
        Example:
            >>> modem = AnalogModem()
            >>> symbols = torch.randn(4, 2, 16, 8, 8)
            >>> tx_sig = modem(symbols)
            >>> # Verify unit power: torch.mean(tx_sig[0]**2) ≈ 1.0
        
        Note:
            Power normalization is applied per sample in the batch,
            not globally across the entire batch.
        """
        # Apply power normalization: E[x^2] = 1
        # Shape: (B, 2, k, H', W') -> (B, 2, k, H', W')
        return self.power_normalize(x)
    
    def demodulate(self, y: torch.Tensor) -> torch.Tensor:
        """
        Demodulate received signal (Rx side).
        
        For analog JSCC, this is a **pass-through (identity) operation**.
        The received noisy signal is directly passed to the decoder without
        any processing, as the neural network decoder is trained end-to-end
        to handle noise and amplitude variations adaptively.
        
        Args:
            y: Received signal tensor of shape (B, 2, k, H', W') from channel.
               This contains noise and/or fading effects.
        
        Returns:
            The same tensor y, unchanged (identity mapping).
        
        Shape:
            - Input: (B, 2, k, H', W')
            - Output: (B, 2, k, H', W')
        
        Example:
            >>> modem = AnalogModem()
            >>> rx_sig = torch.randn(4, 2, 16, 8, 8)
            >>> demod_sig = modem.demodulate(rx_sig)
            >>> torch.equal(rx_sig, demod_sig)  # True
        
        Note:
            Why pass-through?
            - The decoder is trained to implicitly denoise and adapt to
              channel effects. Explicit equalization or de-normalization
              may disrupt the learned representations.
            - For digital JSCC or OFDM schemes, override this method to
              perform FFT, LLR (Log-Likelihood Ratio) calculation, or
              other demodulation steps.
        
        TODO:
            For OFDM or digital modulation schemes, implement:
            - IFFT/FFT operations for OFDM
            - LLR calculation for soft-decision decoding
            - Channel equalization (e.g., ZF, MMSE)
        """
        # Identity operation: directly return received signal
        return y
        

        

