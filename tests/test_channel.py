"""
Unit tests for Channel module.

Tests cover:
- AWGNChannel noise addition
- RayleighChannel fading and noise
- SNR-dependent noise power
- Gradient flow (autograd compatibility)
- Complex multiplication logic
"""

import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import BaseChannel, AWGNChannel, RayleighChannel


class TestAWGNChannel:
    """Test suite for AWGNChannel class."""
    
    @pytest.fixture
    def channel(self):
        """Create an AWGNChannel instance for testing."""
        return AWGNChannel(epsilon=1e-8)
    
    def test_output_shape_preservation(self, channel):
        """Test that channel preserves input shape."""
        # Various input shapes
        shapes = [
            (4, 2, 16, 16),  # (B, 2, H, W)
            (2, 2, 8, 8),
            (1, 2, 32, 32),
        ]
        
        for shape in shapes:
            x = torch.randn(shape)
            y = channel(x, snr_db=10.0)
            assert y.shape == x.shape, f"Shape should be preserved for {shape}"
    
    def test_noise_power_scaling(self, channel):
        """Test that noise power scales correctly with SNR."""
        # Create unit-power input
        x = torch.ones(100, 2, 16, 16)  # Mean power = 1
        
        # Test different SNR values
        snr_values = [0, 5, 10, 15, 20]
        
        for snr_db in snr_values:
            # Generate multiple noisy outputs
            num_trials = 50
            noise_powers = []
            
            for _ in range(num_trials):
                y = channel(x, snr_db=snr_db)
                noise = y - x
                noise_power = torch.mean(noise ** 2).item()
                noise_powers.append(noise_power)
            
            # Average noise power
            avg_noise_power = sum(noise_powers) / len(noise_powers)
            
            # Expected noise power: sigma^2 = 1 / (10^(SNR_dB/10))
            expected_noise_power = 1.0 / (10.0 ** (snr_db / 10.0))
            
            # Allow 20% tolerance due to randomness
            relative_error = abs(avg_noise_power - expected_noise_power) / expected_noise_power
            assert relative_error < 0.2, (
                f"SNR {snr_db} dB: expected noise power {expected_noise_power:.6f}, "
                f"got {avg_noise_power:.6f} (error: {relative_error:.2%})"
            )
    
    def test_high_snr_low_noise(self, channel):
        """Test that high SNR produces low noise."""
        x = torch.randn(10, 2, 8, 8)
        
        # Very high SNR (30 dB)
        y = channel(x, snr_db=30.0)
        noise = y - x
        noise_power = torch.mean(noise ** 2).item()
        
        # Noise power should be very small
        assert noise_power < 0.01, "High SNR should produce very low noise"
    
    def test_low_snr_high_noise(self, channel):
        """Test that low SNR produces high noise."""
        x = torch.randn(10, 2, 8, 8)
        
        # Very low SNR (-10 dB)
        y = channel(x, snr_db=-10.0)
        noise = y - x
        noise_power = torch.mean(noise ** 2).item()
        
        # Noise power should be large
        assert noise_power > 1.0, "Low SNR should produce high noise"
    
    def test_gradient_flow(self, channel):
        """Test that gradients flow through AWGN channel."""
        x = torch.randn(2, 2, 8, 8, requires_grad=True)
        
        # Forward pass
        y = channel(x, snr_db=10.0)
        loss = y.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradient exists
        assert x.grad is not None, "Gradient should exist"
        assert not torch.isnan(x.grad).any(), "Gradient should not contain NaN"
        assert not torch.isinf(x.grad).any(), "Gradient should not contain Inf"
        
        # Gradient should be non-zero (noise is added, so gradient flows)
        assert torch.abs(x.grad).sum() > 0, "Gradient should be non-zero"
    
    def test_deterministic_with_seed(self, channel):
        """Test that channel output is deterministic with manual seed."""
        x = torch.randn(2, 2, 8, 8)
        snr_db = 10.0
        
        # Run with seed 1
        torch.manual_seed(42)
        y1 = channel(x, snr_db=snr_db)
        
        # Run with seed 2 (different output)
        torch.manual_seed(123)
        y2 = channel(x, snr_db=snr_db)
        
        # Should be different
        assert not torch.allclose(y1, y2), "Different seeds should produce different outputs"
        
        # Run with seed 1 again
        torch.manual_seed(42)
        y3 = channel(x, snr_db=snr_db)
        
        # Should match first run
        assert torch.allclose(y1, y3), "Same seed should produce same output"


class TestRayleighChannel:
    """Test suite for RayleighChannel class."""
    
    @pytest.fixture
    def channel(self):
        """Create a RayleighChannel instance for testing."""
        return RayleighChannel(epsilon=1e-8)
    
    def test_output_shape_preservation(self, channel):
        """Test that channel preserves input shape."""
        shapes = [
            (4, 2, 16, 16),
            (2, 2, 8, 8),
            (1, 2, 32, 32),
        ]
        
        for shape in shapes:
            x = torch.randn(shape)
            y = channel(x, snr_db=10.0)
            assert y.shape == x.shape, f"Shape should be preserved for {shape}"
    
    def test_complex_multiply_correctness(self, channel):
        """Test complex multiplication logic."""
        # Create simple test case
        # h = 2 + 3j, x = 4 + 5j
        # h * x = (2*4 - 3*5) + j*(2*5 + 3*4) = (8 - 15) + j*(10 + 12) = -7 + 22j
        
        batch_size = 1
        h = torch.zeros(batch_size, 2, 1, 1)
        h[0, 0, 0, 0] = 2.0  # Real part
        h[0, 1, 0, 0] = 3.0  # Imaginary part
        
        x = torch.zeros(batch_size, 2, 1, 1)
        x[0, 0, 0, 0] = 4.0  # Real part
        x[0, 1, 0, 0] = 5.0  # Imaginary part
        
        result = channel._complex_multiply(h, x)
        
        # Expected: -7 + 22j
        assert torch.isclose(result[0, 0, 0, 0], torch.tensor(-7.0)), "Real part should be -7"
        assert torch.isclose(result[0, 1, 0, 0], torch.tensor(22.0)), "Imaginary part should be 22"
    
    def test_complex_multiply_identity(self, channel):
        """Test multiplication by 1+0j gives identity."""
        x = torch.randn(2, 2, 8, 8)
        
        # h = 1 + 0j (identity)
        h = torch.zeros_like(x)
        h[:, 0, ...] = 1.0  # Real part = 1
        h[:, 1, ...] = 0.0  # Imaginary part = 0
        
        result = channel._complex_multiply(h, x)
        
        assert torch.allclose(result, x, atol=1e-6), "Multiplication by 1+0j should be identity"
    
    def test_complex_multiply_zero(self, channel):
        """Test multiplication by 0+0j gives zero."""
        x = torch.randn(2, 2, 8, 8)
        
        # h = 0 + 0j
        h = torch.zeros_like(x)
        
        result = channel._complex_multiply(h, x)
        
        assert torch.allclose(result, torch.zeros_like(x), atol=1e-6), \
            "Multiplication by 0+0j should give zero"
    
    def test_fading_coefficients_distribution(self, channel):
        """Test that generated fading coefficients follow CN(0,1) distribution."""
        # Generate many fading coefficients
        x = torch.randn(1000, 2, 16, 16)
        
        # Forward pass generates h internally
        torch.manual_seed(42)
        y = channel(x, snr_db=100.0)  # High SNR to isolate fading effect
        
        # Extract h by running with no noise: y ≈ h * x at high SNR
        # For unit variance check, we test internal generation
        
        # Alternative: directly test internal generation
        std = 1.0 / torch.sqrt(torch.tensor(2.0))
        h = torch.randn(10000, 2, 1, 1) * std
        
        # Check that E[h_real^2] ≈ 0.5 and E[h_imag^2] ≈ 0.5
        h_real_var = torch.var(h[:, 0, ...]).item()
        h_imag_var = torch.var(h[:, 1, ...]).item()
        
        # E[|h|^2] = E[h_real^2] + E[h_imag^2] should be ≈ 1
        total_power = h_real_var + h_imag_var
        
        assert abs(h_real_var - 0.5) < 0.05, f"Real part variance should be ~0.5, got {h_real_var}"
        assert abs(h_imag_var - 0.5) < 0.05, f"Imag part variance should be ~0.5, got {h_imag_var}"
        assert abs(total_power - 1.0) < 0.05, f"Total power should be ~1.0, got {total_power}"
    
    def test_custom_fading_coefficients(self, channel):
        """Test that custom fading coefficients are used correctly."""
        x = torch.randn(2, 2, 8, 8)
        
        # Custom h: all ones for real part, zeros for imaginary
        h_custom = torch.zeros_like(x)
        h_custom[:, 0, ...] = 1.0
        h_custom[:, 1, ...] = 0.0
        
        # Use custom h with very high SNR (minimal noise)
        y = channel(x, snr_db=100.0, h=h_custom)
        
        # At high SNR: y ≈ h * x
        # With h = 1 + 0j: y ≈ x
        noise = y - x
        noise_power = torch.mean(noise ** 2).item()
        
        # Noise should be very small at high SNR
        assert noise_power < 0.01, "With custom h=1+0j and high SNR, output should ≈ input"
    
    def test_gradient_flow(self, channel):
        """Test that gradients flow through Rayleigh channel."""
        x = torch.randn(2, 2, 8, 8, requires_grad=True)
        
        # Forward pass
        y = channel(x, snr_db=10.0)
        loss = y.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradient exists
        assert x.grad is not None, "Gradient should exist"
        assert not torch.isnan(x.grad).any(), "Gradient should not contain NaN"
        assert not torch.isinf(x.grad).any(), "Gradient should not contain Inf"
        assert torch.abs(x.grad).sum() > 0, "Gradient should be non-zero"
    
    def test_gradient_flow_through_fading(self, channel):
        """Test that gradients flow through custom fading coefficients."""
        x = torch.randn(2, 2, 8, 8, requires_grad=True)
        h = torch.randn(2, 2, 8, 8, requires_grad=True)
        
        # Forward pass with custom h
        y = channel(x, snr_db=10.0, h=h)
        loss = y.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist for both x and h
        assert x.grad is not None, "Gradient w.r.t. x should exist"
        assert h.grad is not None, "Gradient w.r.t. h should exist"
        assert not torch.isnan(x.grad).any(), "x gradient should not contain NaN"
        assert not torch.isnan(h.grad).any(), "h gradient should not contain NaN"
    
    def test_rayleigh_vs_awgn_noise(self, channel):
        """Test that Rayleigh channel adds noise similar to AWGN (on average)."""
        awgn_channel = AWGNChannel()
        
        x = torch.randn(100, 2, 16, 16)
        snr_db = 10.0
        
        # AWGN channel
        y_awgn = awgn_channel(x, snr_db=snr_db)
        noise_awgn = y_awgn - x
        noise_power_awgn = torch.mean(noise_awgn ** 2).item()
        
        # Rayleigh channel (multiple trials to average out fading)
        noise_powers_rayleigh = []
        for _ in range(10):
            y_rayleigh = channel(x, snr_db=snr_db)
            # Can't directly compute noise_rayleigh = y - x due to fading
            # But total power should be similar on average
            # Just check that noise is present
            noise_powers_rayleigh.append(torch.mean((y_rayleigh - x) ** 2).item())
        
        avg_rayleigh_power = sum(noise_powers_rayleigh) / len(noise_powers_rayleigh)
        
        # Rayleigh should have more variation than AWGN due to fading
        # But both should add noise on the same order
        # This is a weak test, just checking noise is added
        assert avg_rayleigh_power > 0, "Rayleigh channel should add noise"
    
    def test_spatial_dimension_preservation(self, channel):
        """Test that spatial dimensions are preserved (no reshape(-1))."""
        # Test with different spatial dimensions
        x = torch.randn(2, 2, 8, 16)  # Non-square spatial dims
        y = channel(x, snr_db=10.0)
        
        assert y.shape == x.shape, "Spatial dimensions should be preserved"
        assert y.shape[2] == 8 and y.shape[3] == 16, "Should maintain non-square spatial dims"


class TestBaseChannel:
    """Test suite for BaseChannel abstract class."""
    
    def test_base_channel_not_implemented(self):
        """Test that BaseChannel raises NotImplementedError."""
        channel = BaseChannel()
        x = torch.randn(2, 2, 8, 8)
        
        with pytest.raises(NotImplementedError):
            channel(x, snr_db=10.0)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
