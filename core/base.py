"""
Base Classes for SemCom_Infra.

This module defines the abstract base classes and system-level wrapper for
the semantic communication pipeline:
- BaseEncoder: Abstract encoder (Tx) that maps images to semantic symbols.
- BaseDecoder: Abstract decoder (Rx) that recovers images from noisy symbols.
- JSCCSystem: Concrete wrapper that chains Encoder -> Modem -> Channel -> Decoder.

Data Flow Convention:
    Image (B, C, H, W)
      -> Encoder -> (B, 2, k, H', W')   [semantic symbols, dim=1 = I/Q]
      -> Modem.power_normalize -> (B, 2, k, H', W')  [unit power]
      -> Channel -> (B, 2, k, H', W')   [noisy symbols]
      -> Decoder -> (B, C, H, W)        [reconstructed image]

Note:
    - All intermediate signals use (B, 2, ...) where dim=1 is [I, Q].
    - This convention is enforced by BaseEncoder (output) and BaseDecoder (input).
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from core.channel import BaseChannel
from core.modem import BaseModem


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for Semantic Encoders (Tx).

    The encoder maps input images to semantic symbols suitable for
    transmission over a physical channel. The output tensor follows
    the (B, 2, k, H', W') convention where dim=1 is [I, Q].

    Subclass Implementation Guide:
        1. Build a CNN (or Transformer) backbone that outputs (B, 2*k, H', W').
        2. In forward(), after the last Conv2d, **reshape** the output:
           ``x = x.view(B, 2, k, H', W')``
           so that dim=1 corresponds to Real/Imag components.

    Note:
        - Output is NOT power-normalized; normalization is done by Modem.
        - The last Conv2d layer should have ``out_channels = 2 * k``,
          where k is the number of complex symbols per spatial location.
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode input image into semantic symbols.

        Args:
            x: Input image tensor of shape (B, C, H, W).
            **kwargs: Reserved keyword arguments for future extensions:
                - rate_idx (Optional[int]): Index for dynamic rate selection.
                  Used by WITT / NTSCC to support variable bandwidth ratios.
                - quantizer (Optional[nn.Module]): Quantization module for
                  Digital JSCC schemes.

        Returns:
            Semantic symbols of shape (B, 2, k, H', W'), where:
                - dim=1 index 0 -> In-phase (I) component
                - dim=1 index 1 -> Quadrature (Q) component
                - k is the number of complex symbols per spatial location

        Shape:
            - Input: (B, C, H, W)
            - Output: (B, 2, k, H', W')

        Note:
            The output is un-normalized. Power normalization is handled
            by the Modem module downstream.

        Example (subclass implementation sketch)::

            def forward(self, x, **kwargs):
                # CNN backbone: (B, C, H, W) -> (B, 2*k, H', W')
                z = self.backbone(x)
                B = z.shape[0]
                # Reshape to channel convention: (B, 2*k, H', W') -> (B, 2, k, H', W')
                z = z.view(B, 2, -1, z.shape[2], z.shape[3])
                return z
        """
        ...


class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for Semantic Decoders (Rx).

    The decoder recovers images from received (noisy) semantic symbols.
    Input follows the (B, 2, k, H', W') convention from the channel output.

    Subclass Implementation Guide:
        1. In forward(), **flatten** the input first:
           ``x = x.view(B, 2*k, H', W')``
           to produce a standard feature map for Conv2d / ConvTranspose2d.
        2. Build a mirrored CNN (ConvTranspose2d) backbone to upsample
           back to the original image resolution.
        3. Use ``Sigmoid`` as the final activation to output pixel values
           in [0, 1].

    Note:
        - The decoder is structurally the mirror/inverse of the encoder.
        - V1 returns a single Tensor (reconstructed image).
        - Future versions may return a Dict with multiple outputs
          (e.g., task predictions, CSI estimates).
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode received symbols into reconstructed image.

        Args:
            x: Received symbol tensor of shape (B, 2, k, H', W').
               This comes from the channel output (noisy symbols).
            **kwargs: Reserved keyword arguments for future extensions:
                - csi (Optional[torch.Tensor]): Channel State Information
                  for receiver-side guidance / equalization.

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]:
                - V1: Reconstructed image tensor of shape (B, C, H, W),
                  with pixel values in [0, 1] (Sigmoid output).
                - Future: Dict containing multiple outputs, e.g.::

                    {
                        "reconstruction": torch.Tensor,  # (B, C, H, W)
                        "task_output": torch.Tensor,      # task-specific head
                        "csi_estimation": torch.Tensor,   # estimated CSI
                    }

        Shape:
            - Input: (B, 2, k, H', W')
            - Output: (B, C, H, W)

        Example (subclass implementation sketch)::

            def forward(self, x, **kwargs):
                B = x.shape[0]
                # Flatten I/Q and symbols: (B, 2, k, H', W') -> (B, 2*k, H', W')
                x = x.view(B, -1, x.shape[3], x.shape[4])
                # ConvTranspose2d backbone: (B, 2*k, H', W') -> (B, C, H, W)
                out = self.backbone(x)
                # Final activation: pixel values in [0, 1]
                out = torch.sigmoid(out)
                return out
        """
        ...


class JSCCSystem(nn.Module):
    """
    Joint Source-Channel Coding System Wrapper.

    This module chains the full semantic communication pipeline:
        Encoder -> [Quantizer] -> Modem -> Channel -> [Equalizer] -> Decoder

    It serves as the top-level nn.Module for training and inference.

    Args:
        encoder: Semantic encoder mapping images to symbols.
        decoder: Semantic decoder recovering images from noisy symbols.
        modem: Modem for power normalization (and future OFDM).
        channel: Differentiable channel model (AWGN, Rayleigh, etc.).

    Example:
        >>> encoder = MyEncoder(in_channels=3, symbol_channels=16)
        >>> decoder = MyDecoder(out_channels=3, symbol_channels=16)
        >>> modem = BaseModem()
        >>> channel = AWGNChannel()
        >>> system = JSCCSystem(encoder, decoder, modem, channel)
        >>> img = torch.randn(4, 3, 32, 32)
        >>> recon = system(img, snr_db=10.0)
        >>> recon.shape  # (4, 3, 32, 32)
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        modem: BaseModem,
        channel: BaseChannel,
    ) -> None:
        """
        Initialize JSCCSystem.

        Args:
            encoder: Semantic encoder (BaseEncoder subclass).
            decoder: Semantic decoder (BaseDecoder subclass).
            modem: Modem for power normalization (BaseModem subclass).
            channel: Differentiable channel (BaseChannel subclass).
        """
        super(JSCCSystem, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.modem = modem
        self.channel = channel

    def forward(
        self,
        img: torch.Tensor,
        snr_db: float,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the full JSCC pipeline: encode -> normalize -> channel -> decode.

        Args:
            img: Input image tensor of shape (B, C, H, W),
                 with pixel values in [0, 1].
            snr_db: Channel Signal-to-Noise Ratio in dB.
            **kwargs: Additional keyword arguments forwarded to encoder/decoder.
                Supported keys include:
                - rate_idx: Dynamic rate index (for encoder).
                - csi: Channel state information (for decoder).

        Returns:
            Reconstructed image tensor of shape (B, C, H, W),
            with pixel values in [0, 1].

        Pipeline:
            1. Encode:    symbols = encoder(img)           -> (B, 2, k, H', W')
            2. Quantize:  (reserved)
            3. Normalize: tx_sig = modem.power_normalize() -> (B, 2, k, H', W')
            4. Channel:   rx_sig = channel(tx_sig, snr_db) -> (B, 2, k, H', W')
            5. Equalize:  (reserved)
            6. Decode:    out = decoder(rx_sig)             -> (B, C, H, W)
        """
        # --- Step 1: Semantic Encoding ---
        # (B, C, H, W) -> (B, 2, k, H', W')
        symbols = self.encoder(img, **kwargs)

        # --- Step 2: Digital Quantization (reserved) ---
        # TODO: Implement Digital Quantization here (for Digital JSCC / finite-bit schemes)
        # Example: symbols = quantizer(symbols) if kwargs.get('quantizer') else symbols

        # --- Step 3: Power Normalization ---
        # Ensure E[x^2] = 1 before transmission
        # (B, 2, k, H', W') -> (B, 2, k, H', W')
        tx_sig = self.modem.power_normalize(symbols)

        # --- Step 4: Channel Transmission ---
        # Add noise / fading based on SNR
        # (B, 2, k, H', W') -> (B, 2, k, H', W')
        rx_sig = self.channel(tx_sig, snr_db)

        # --- Step 5: Channel Equalization (reserved) ---
        # TODO: Implement Equalizer / CSI estimation here
        # Example: rx_sig, csi_hat = equalizer(rx_sig, h) if equalizer else (rx_sig, None)
        # Pass estimated CSI to decoder if available:
        # kwargs['csi'] = csi_hat

        # --- Step 6: Semantic Decoding ---
        # (B, 2, k, H', W') -> (B, C, H, W)
        out = self.decoder(rx_sig, **kwargs)

        return out
