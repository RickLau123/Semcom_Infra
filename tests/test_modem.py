"""
Unit tests for Modem module.

Tests cover:
- Power normalization
- Complex <-> Real conversion
- Gradient flow (autograd compatibility)
"""

import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.modem import BaseModem


class TestBaseModem:
    """Test suite for BaseModem class."""
    
    @pytest.fixture
    def modem(self):
        """Create a BaseModem instance for testing."""
        return BaseModem(epsilon=1e-8)
    
    def test_power_normalize_unit_power(self, modem):
        """Test that power_normalize produces signals with unit power."""
        # Create random input
        batch_size = 4
        x = torch.randn(batch_size, 2, 16, 16)
        
        # Normalize
        x_norm = modem.power_normalize(x)
        
        # Check shape is preserved
        assert x_norm.shape == x.shape, "Shape should be preserved"
        
        # Check power is approximately 1 for each sample
        for i in range(batch_size):
            sample = x_norm[i]
            power = torch.mean(sample ** 2).item()
            assert abs(power - 1.0) < 0.01, f"Sample {i} power should be ~1.0, got {power}"
    
    def test_power_normalize_batch_independence(self, modem):
        """Test that each sample in batch is normalized independently."""
        # Create inputs with different scales
        x1 = torch.randn(1, 2, 8, 8) * 0.5  # Low power
        x2 = torch.randn(1, 2, 8, 8) * 2.0  # High power
        x = torch.cat([x1, x2], dim=0)  # (2, 2, 8, 8)
        
        # Normalize
        x_norm = modem.power_normalize(x)
        
        # Both samples should have unit power
        power1 = torch.mean(x_norm[0] ** 2).item()
        power2 = torch.mean(x_norm[1] ** 2).item()
        
        assert abs(power1 - 1.0) < 0.01, "Sample 1 should have unit power"
        assert abs(power2 - 1.0) < 0.01, "Sample 2 should have unit power"
        assert abs(power1 - power2) < 0.01, "Both samples should have same power"
    
    def test_power_normalize_gradient_flow(self, modem):
        """Test that gradients flow through power_normalize."""
        x = torch.randn(2, 2, 8, 8, requires_grad=True)
        
        # Forward pass
        x_norm = modem.power_normalize(x)
        loss = x_norm.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradient exists
        assert x.grad is not None, "Gradient should exist"
        assert not torch.isnan(x.grad).any(), "Gradient should not contain NaN"
        assert not torch.isinf(x.grad).any(), "Gradient should not contain Inf"
    
    def test_power_normalize_zero_handling(self, modem):
        """Test that power_normalize handles near-zero inputs gracefully."""
        # Very small input (near zero)
        x = torch.randn(2, 2, 8, 8) * 1e-10
        
        # Should not raise error or produce NaN/Inf
        x_norm = modem.power_normalize(x)
        
        assert not torch.isnan(x_norm).any(), "Should not produce NaN"
        assert not torch.isinf(x_norm).any(), "Should not produce Inf"
    
    def test_complex_to_real_conversion(self):
        """Test complex to real tensor conversion."""
        # Create complex tensor
        batch_size, channels, height, width = 4, 3, 8, 8
        x_complex = torch.randn(batch_size, channels, height, width, dtype=torch.complex64)
        
        # Convert to real
        x_real = BaseModem.complex_to_real(x_complex)
        
        # Check shape: (B, C, H, W) -> (B, 2, C, H, W)
        expected_shape = (batch_size, 2, channels, height, width)
        assert x_real.shape == expected_shape, f"Expected shape {expected_shape}, got {x_real.shape}"
        
        # Check dtype is real
        assert x_real.dtype in [torch.float32, torch.float64], "Output should be real-valued"
        
        # Verify values match
        assert torch.allclose(x_real[:, 0, ...], x_complex.real), "Real parts should match"
        assert torch.allclose(x_real[:, 1, ...], x_complex.imag), "Imaginary parts should match"
    
    def test_real_to_complex_conversion(self):
        """Test real to complex tensor conversion."""
        # Create real tensor with shape (B, 2, C, H, W)
        batch_size, channels, height, width = 4, 3, 8, 8
        x_real = torch.randn(batch_size, 2, channels, height, width)
        
        # Convert to complex
        x_complex = BaseModem.real_to_complex(x_real)
        
        # Check shape: (B, 2, C, H, W) -> (B, C, H, W)
        expected_shape = (batch_size, channels, height, width)
        assert x_complex.shape == expected_shape, f"Expected shape {expected_shape}, got {x_complex.shape}"
        
        # Check dtype is complex
        assert x_complex.dtype in [torch.complex64, torch.complex128], "Output should be complex"
        
        # Verify values match
        assert torch.allclose(x_complex.real, x_real[:, 0, ...]), "Real parts should match"
        assert torch.allclose(x_complex.imag, x_real[:, 1, ...]), "Imaginary parts should match"
    
    def test_complex_real_roundtrip(self):
        """Test that complex -> real -> complex is identity."""
        # Create complex tensor
        x_complex_original = torch.randn(2, 3, 8, 8, dtype=torch.complex64)
        
        # Convert: complex -> real -> complex
        x_real = BaseModem.complex_to_real(x_complex_original)
        x_complex_reconstructed = BaseModem.real_to_complex(x_real)
        
        # Should reconstruct original
        assert torch.allclose(x_complex_original, x_complex_reconstructed, atol=1e-6), \
            "Roundtrip conversion should preserve values"
    
    def test_real_complex_roundtrip(self):
        """Test that real -> complex -> real is identity."""
        # Create real tensor
        x_real_original = torch.randn(2, 2, 3, 8, 8)
        
        # Convert: real -> complex -> real
        x_complex = BaseModem.real_to_complex(x_real_original)
        x_real_reconstructed = BaseModem.complex_to_real(x_complex)
        
        # Should reconstruct original
        assert torch.allclose(x_real_original, x_real_reconstructed, atol=1e-6), \
            "Roundtrip conversion should preserve values"
    
    def test_real_to_complex_invalid_shape(self):
        """Test that real_to_complex raises error for invalid shape."""
        # Create tensor with wrong dim=1 size
        x_invalid = torch.randn(2, 3, 8, 8)  # dim=1 has size 3, not 2
        
        # Should raise assertion error
        with pytest.raises(AssertionError):
            BaseModem.real_to_complex(x_invalid)
    
    def test_complex_to_real_invalid_dtype(self):
        """Test that complex_to_real raises error for non-complex input."""
        # Create real tensor (not complex)
        x_real = torch.randn(2, 3, 8, 8)
        
        # Should raise assertion error
        with pytest.raises(AssertionError):
            BaseModem.complex_to_real(x_real)
    
    def test_conversion_gradient_flow(self):
        """Test that gradients flow through complex/real conversions."""
        # Start with complex tensor
        x_complex = torch.randn(2, 3, 8, 8, dtype=torch.complex64)
        
        # Wrap in real tensor with requires_grad
        x_real = BaseModem.complex_to_real(x_complex)
        x_real.requires_grad_(True)
        
        # Forward: real -> complex -> operation
        x_complex_out = BaseModem.real_to_complex(x_real)
        loss = (x_complex_out.abs() ** 2).sum()
        
        # Backward
        loss.backward()
        
        # Check gradient exists
        assert x_real.grad is not None, "Gradient should exist"
        assert not torch.isnan(x_real.grad).any(), "Gradient should not contain NaN"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
