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
