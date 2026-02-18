"""
End-to-End Semantic Communication System for SemCom_Infra.

This module defines :class:`SemComSystem`, which connects all components
of a semantic communication pipeline into a single differentiable
``nn.Module``:

    Image -> Encoder -> Modem (Tx) -> Channel -> Modem (Rx) -> Decoder -> Image_hat

Data Flow:
    Image (B, C, H, W)
      -> Encoder     -> (B, 2, k, H', W')   [semantic symbols]
      -> Modem.forward (Tx)  -> (B, 2, k, H', W')   [power-normalised]
      -> Channel     -> (B, 2, k, H', W')   [noisy symbols]
      -> Modem.demodulate (Rx) -> (B, 2, k, H', W') [demodulated]
      -> Decoder     -> (B, C, H, W)        [reconstructed image]

Example:
    >>> from core.codec import CNNEncoder, CNNDecoder, build_deepjscc_config
    >>> from core.modem import AnalogModem
    >>> from core.channel import AWGNChannel
    >>>
    >>> cfg = build_deepjscc_config(cbr=1/6)
    >>> system = SemComSystem(
    ...     encoder=CNNEncoder(**cfg['encoder']),
    ...     decoder=CNNDecoder(**cfg['decoder']),
    ...     modem=AnalogModem(),
    ...     channel=AWGNChannel(),
    ... )
    >>> x = torch.randn(4, 3, 32, 32).clamp(0, 1)
    >>> x_hat = system(x, snr_db=10.0)
    >>> x_hat.shape  # (4, 3, 32, 32)
"""

import torch
import torch.nn as nn
from typing import Any, Dict

from core.codec import BaseEncoder, BaseDecoder
from core.modem import BaseModem
from core.channel import BaseChannel
from core.utils import tensor_to_image


class SemComSystem(nn.Module):
    """
    End-to-end Semantic Communication System.

    Connects encoder, modem, channel, and decoder into a single
    differentiable pipeline for joint source-channel coding training.

    Args:
        encoder: Semantic encoder (Tx) that maps images to symbols.
        decoder: Semantic decoder (Rx) that reconstructs images from
            noisy symbols.
        modem: Modem module for power normalisation and (de)modulation.
        channel: Differentiable channel model (e.g., AWGN, Rayleigh).
                **kwargs: Extra modules or configs for future extensions.

                        - If value is ``nn.Module``, it will be registered to
                            ``self.custom_modules`` (so parameters are tracked).
                        - Otherwise, it will be stored in ``self.custom_config``.

    Example::

        system = SemComSystem(encoder, decoder, modem, channel)
        x_hat = system(x, snr_db=10.0)
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        modem: BaseModem,
        channel: BaseChannel,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.modem = modem
        self.channel = channel

        # Flexible extension entry:
        # - custom_modules: tracked by state_dict / optimizer
        # - custom_config: plain Python configs
        self.custom_modules = nn.ModuleDict()
        self.custom_config: Dict[str, Any] = {}

        for key, value in kwargs.items():
            if isinstance(value, nn.Module):
                self.custom_modules[key] = value
            else:
                self.custom_config[key] = value

    # ------------------------------------------------------ custom utilities
    def has_custom_module(self, name: str) -> bool:
        """Check whether a custom module with given name exists."""
        return name in self.custom_modules

    def get_custom_module(self, name: str) -> nn.Module:
        """Get a registered custom module by name."""
        if name not in self.custom_modules:
            raise KeyError(f"Custom module '{name}' is not registered.")
        return self.custom_modules[name]

    def get_custom_config(self, name: str, default: Any = None) -> Any:
        """Get a custom config value by name."""
        return self.custom_config.get(name, default)

    # ----------------------------------------------------------- step methods
    def preprocess_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optional preprocessing step before encoder.

        Override this method when you want to add image-side innovation
        modules (e.g., denoiser / feature adaptor) without rewriting
        the whole ``forward``.
        """
        return x

    def encode_step(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to semantic symbols."""
        return self.encoder(x)

    def tx_step(self, z: torch.Tensor) -> torch.Tensor:
        """Transmit-side modem step (power normalization/modulation)."""
        return self.modem(z)

    def channel_step(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Differentiable channel propagation step."""
        return self.channel(z, snr_db)

    def rx_step(self, z: torch.Tensor) -> torch.Tensor:
        """Receive-side modem step (demodulation)."""
        return self.modem.demodulate(z)

    def decode_step(self, z: torch.Tensor) -> torch.Tensor:
        """Decode semantic symbols to reconstructed image."""
        return self.decoder(z)

    def postprocess_step(self, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Optional post-processing step after decoder.

        Keep output in ``[0, 1]`` for training compatibility.
        Override this method for learned enhancement modules.
        """
        return x_hat
        # TODO: whether plus 255 to map its value into [0, 255] or not ?

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        End-to-end forward pass through the semantic communication system.

        Args:
            x: Input image tensor of shape ``(B, C, H, W)``,
                with pixel values in ``[0, 1]``.
            snr_db: Channel SNR in dB (scalar).

        Returns:
            Reconstructed image tensor of shape ``(B, C, H, W)``,
            with pixel values in ``[0, 1]``.

        Shape:
            - Input:  ``(B, C, H, W)``
            - Output: ``(B, C, H, W)``
        """
        # Optional preprocess before encoder
        # (B, C, H, W) -> (B, C, H, W)
        x = self.preprocess_step(x)

        # Encode: (B, C, H, W) -> (B, 2, k, H', W')
        z = self.encode_step(x)

        # Tx modem: (B, 2, k, H', W') -> (B, 2, k, H', W')
        z = self.tx_step(z)

        # Channel: (B, 2, k, H', W') -> (B, 2, k, H', W')
        z = self.channel_step(z, snr_db)

        # Rx modem: (B, 2, k, H', W') -> (B, 2, k, H', W')
        z = self.rx_step(z)

        # Decode: (B, 2, k, H', W') -> (B, C, H, W)
        x_hat = self.decode_step(z)

        # Optional postprocess after decoder, keep in [0,1]
        # (B, C, H, W) -> (B, C, H, W)
        x_hat = self.postprocess_step(x_hat)

        return x_hat

    # ----------------------------------------------------------- reconstruct
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Reconstruct images for visualisation (no gradient tracking).

        Runs the full pipeline and converts the output to ``[0, 255]``
        pixel range via :func:`core.utils.tensor_to_image`.

        Args:
            x: Input image tensor of shape ``(B, C, H, W)``,
                values in ``[0, 1]``.
            snr_db: Channel SNR in dB.

        Returns:
            Image tensor with pixel values in ``[0, 255]``,
            shape ``(B, C, H, W)``.
        """
        self.eval()
        x_hat = self.forward(x, snr_db)
        return tensor_to_image(x_hat)
