"""
Base Channel Module for SemCom_Infra.

This module provides the abstract base class for differentiable channel models:
- AWGNChannel: Additive White Gaussian Noise channel
- RayleighChannel: Rayleigh fading channel with AWGN

All channels operate on real-valued tensors with shape (B, 2, ...) where
dim=1 contains [I, Q] (in-phase and quadrature components).
"""

import torch
import torch.nn as nn
from typing import Optional


class BaseChannel(nn.Module):
    """
    Abstract base class for Channel modules.
    
    The Channel is responsible for:
    1. Adding noise based on SNR (Signal-to-Noise Ratio)
    2. Simulating physical channel effects (e.g., fading)
    3. Maintaining differentiability for backpropagation
    
    Note:
        - Input signals must have power normalized to 1 (E[x^2] = 1)
        - SNR is provided in dB and converted to linear scale internally
        - All operations must support PyTorch autograd (no detach)
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize BaseChannel.
        
        Args:
            epsilon: Small constant to prevent numerical instability.
                    Default: 1e-8
        """
        super(BaseChannel, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Forward pass through channel (to be implemented in subclasses).
        
        Args:
            x: Input signal tensor of shape (B, 2, ...), where dim=1 is [I, Q].
            snr_db: Signal-to-Noise Ratio in dB.
        
        Returns:
            Noisy signal tensor of the same shape as input.
        
        Raises:
            NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("Subclasses must implement forward method")


class AWGNChannel(BaseChannel):
    """
    Additive White Gaussian Noise (AWGN) Channel.
    
    This channel adds Gaussian noise to the input signal based on the specified SNR.
    The noise power is calculated assuming the signal power is normalized to 1.
    
    Formula:
        y = x + noise
        where noise ~ N(0, sigma^2 * I)
        and sigma = sqrt(1 / (10^(snr_db/10)))
    
    Note:
        - Assumes input signal has unit power: E[x^2] = 1
        - SNR in dB: SNR_dB = 10 * log10(P_signal / P_noise)
        - For P_signal = 1, we have P_noise = 10^(-SNR_dB/10)
        - Noise std: sigma = sqrt(P_noise) = sqrt(1 / 10^(SNR_dB/10))
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize AWGNChannel.
        
        Args:
            epsilon: Small constant to prevent numerical instability.
                    Default: 1e-8
        """
        super(AWGNChannel, self).__init__(epsilon=epsilon)
    
    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Add AWGN to input signal.
        
        Args:
            x: Input signal tensor of shape (B, 2, H, W) or (B, 2, ...).
               Assumes unit power: E[x^2] = 1.
            snr_db: Signal-to-Noise Ratio in dB.
        
        Returns:
            Noisy signal y = x + noise, same shape as input.
        
        Shape:
            - Input: (B, 2, *) where * means any number of spatial dimensions
            - Output: (B, 2, *) same shape as input
        
        Example:
            >>> channel = AWGNChannel()
            >>> x = torch.randn(4, 2, 16, 16)  # Assume already power normalized
            >>> y = channel(x, snr_db=10.0)
            >>> y.shape  # (4, 2, 16, 16)
        
        Note:
            The operation is fully differentiable and supports autograd.
        """
        # Convert SNR from dB to linear scale
        # SNR_linear = 10^(SNR_dB / 10)
        snr_linear = 10.0 ** (snr_db / 10.0)
        
        # Calculate noise standard deviation
        # For unit signal power: sigma = sqrt(1 / SNR_linear)
        sigma = torch.sqrt(torch.tensor(1.0 / (snr_linear + self.epsilon)))
        
        # Generate Gaussian noise with same shape as input
        # noise ~ N(0, sigma^2)
        noise = torch.randn_like(x) * sigma
        
        # Add noise to signal
        # Shape: (B, 2, ...) + (B, 2, ...) -> (B, 2, ...)
        y = x + noise
        
        return y


class RayleighChannel(BaseChannel):
    """
    Rayleigh Fading Channel with AWGN.
    
    This channel models frequency-flat Rayleigh fading with additive noise.
    Each sample experiences independent fading coefficients.
    
    Formula:
        y = h ⊙ x + noise
        where:
        - h ~ CN(0, 1) is the complex fading coefficient (Rayleigh distributed amplitude)
        - ⊙ denotes complex multiplication
        - noise ~ N(0, sigma^2 * I)
    
    Complex Multiplication (using real tensors):
        If h = h_real + j*h_imag and x = x_real + j*x_imag,
        then h * x = (h_real*x_real - h_imag*x_imag) + j*(h_real*x_imag + h_imag*x_real)
    
    Note:
        - Fading coefficients are generated independently for each sample
        - E[|h|^2] = 1 (unit average fading power)
        - Supports spatial dimensions (fading applied element-wise)
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize RayleighChannel.
        
        Args:
            epsilon: Small constant to prevent numerical instability.
                    Default: 1e-8
        """
        super(RayleighChannel, self).__init__(epsilon=epsilon)
    
    def _complex_multiply(
        self, 
        h: torch.Tensor, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform complex multiplication using real-valued tensors.
        
        Args:
            h: Fading coefficients of shape (B, 2, *), where:
                - h[:, 0, ...] is real part (I)
                - h[:, 1, ...] is imaginary part (Q)
            x: Input signal of shape (B, 2, *), where:
                - x[:, 0, ...] is real part (I)
                - x[:, 1, ...] is imaginary part (Q)
        
        Returns:
            Result of complex multiplication h * x, shape (B, 2, *).
        
        Formula:
            result_real = h_real * x_real - h_imag * x_imag
            result_imag = h_real * x_imag + h_imag * x_real
        
        Shape:
            - h: (B, 2, H, W)
            - x: (B, 2, H, W)
            - output: (B, 2, H, W)
        """
        # Extract real and imaginary components
        # Shape: (B, 2, ...) -> (B, ...)
        h_real = h[:, 0, ...]  # (B, H, W, ...)
        h_imag = h[:, 1, ...]  # (B, H, W, ...)
        x_real = x[:, 0, ...]  # (B, H, W, ...)
        x_imag = x[:, 1, ...]  # (B, H, W, ...)
        
        # Complex multiplication
        # (h_real + j*h_imag) * (x_real + j*x_imag)
        # = (h_real*x_real - h_imag*x_imag) + j*(h_real*x_imag + h_imag*x_real)
        result_real = h_real * x_real - h_imag * x_imag  # (B, H, W, ...)
        result_imag = h_real * x_imag + h_imag * x_real  # (B, H, W, ...)
        
        # Stack back to (B, 2, ...)
        result = torch.stack([result_real, result_imag], dim=1)  # (B, 2, H, W, ...)
        
        return result
    
    def forward(
        self, 
        x: torch.Tensor, 
        snr_db: float,
        h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply Rayleigh fading and AWGN to input signal.
        
        Args:
            x: Input signal tensor of shape (B, 2, H, W) or (B, 2, ...).
               Assumes unit power: E[x^2] = 1.
            snr_db: Signal-to-Noise Ratio in dB.
            h: Optional pre-generated fading coefficients of shape (B, 2, H, W).
               If None, generates random Rayleigh fading coefficients.
               Each coefficient h ~ CN(0, 1), i.e., h_real, h_imag ~ N(0, 0.5).
        
        Returns:
            Faded and noisy signal y = h * x + noise, same shape as input.
        
        Shape:
            - Input x: (B, 2, H, W) or (B, 2, ...)
            - Input h (optional): (B, 2, H, W) or (B, 2, ...)
            - Output: (B, 2, H, W) or (B, 2, ...), same as input
        
        Example:
            >>> channel = RayleighChannel()
            >>> x = torch.randn(4, 2, 16, 16)  # Assume already power normalized
            >>> y = channel(x, snr_db=10.0)
            >>> y.shape  # (4, 2, 16, 16)
            >>> # Or with custom fading:
            >>> h = torch.randn(4, 2, 16, 16) / torch.sqrt(torch.tensor(2.0))
            >>> y = channel(x, snr_db=10.0, h=h)
        
        Note:
            - Fading coefficients h ~ CN(0, 1) satisfy E[|h|^2] = 1
            - For CN(0, 1): h_real ~ N(0, 0.5), h_imag ~ N(0, 0.5)
            - The operation is fully differentiable and supports autograd
        """
        # Generate Rayleigh fading coefficients if not provided
        if h is None:
            # h ~ CN(0, 1): complex Gaussian with unit variance
            # For h = h_real + j*h_imag to satisfy E[|h|^2] = E[h_real^2 + h_imag^2] = 1,
            # we need h_real, h_imag ~ N(0, 0.5)
            # Thus: std = 1/sqrt(2) ≈ 0.7071
            std = 1.0 / torch.sqrt(torch.tensor(2.0))
            h = torch.randn_like(x) * std  # (B, 2, H, W, ...)
        
        # Apply fading: h * x (complex multiplication)
        # Shape: (B, 2, ...) * (B, 2, ...) -> (B, 2, ...)
        x_faded = self._complex_multiply(h, x)
        
        # Convert SNR from dB to linear scale
        snr_linear = 10.0 ** (snr_db / 10.0)
        
        # Calculate noise standard deviation
        # For unit signal power: sigma = sqrt(1 / SNR_linear)
        sigma = torch.sqrt(torch.tensor(1.0 / (snr_linear + self.epsilon)))
        
        # Generate Gaussian noise
        noise = torch.randn_like(x) * sigma
        
        # Add noise: y = h * x + noise
        # Shape: (B, 2, ...) + (B, 2, ...) -> (B, 2, ...)
        y = x_faded + noise
        
        return y
