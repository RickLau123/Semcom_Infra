"""
Codec Module for SemCom_Infra.

This module defines the abstract base classes and concrete CNN implementations
for semantic communication encoders and decoders:

Abstract Base Classes:
    - BaseEncoder: Abstract encoder (Tx) that maps images to semantic symbols.
    - BaseDecoder: Abstract decoder (Rx) that recovers images from noisy symbols.

Concrete CNN Implementations:
    - CNNEncoder: Flexible CNN-based encoder with configurable layer architecture.
    - CNNDecoder: Flexible CNN-based decoder with configurable layer architecture.

Helper Functions:
    - build_deepjscc_config: Generate standard DeepJSCC configuration.

Data Flow Convention:
    Image (B, C, H, W)
      -> Encoder -> (B, 2, k, H', W')   [semantic symbols, dim=1 = I/Q]
      -> Modem.power_normalize -> (B, 2, k, H', W')  [unit power]
      -> Channel -> (B, 2, k, H', W')   [noisy symbols]
      -> Decoder -> (B, C, H, W)        [reconstructed image]

Note:
    - All intermediate signals use (B, 2, ...) where dim=1 is [I, Q].
    - This convention is enforced by BaseEncoder (output) and BaseDecoder (input).
    - The pipeline wrapper (JSCCSystem) is defined separately in core/pipeline.py.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union


# ============================================================================
# Abstract Base Classes
# ============================================================================


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


# ============================================================================
# Helper Functions for Layer Construction
# ============================================================================


def _get_activation(act_type: str) -> nn.Module:
    """
    Get activation layer by name.

    Args:
        act_type: Activation type ('relu', 'prelu', 'leaky_relu').

    Returns:
        Activation module.

    Raises:
        ValueError: If act_type is not recognized.
    """
    act_type = act_type.lower()
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    elif act_type == "prelu":
        return nn.PReLU()
    elif act_type == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        raise ValueError(
            f"Unsupported activation type: {act_type}. "
            f"Choose from 'relu', 'prelu', 'leaky_relu'."
        )


def _get_norm_layer(norm_type: str, num_features: int) -> Union[nn.Module, None]:
    """
    Get normalization layer by name.

    Args:
        norm_type: Normalization type ('none', 'batch', 'instance').
        num_features: Number of features for normalization.

    Returns:
        Normalization module or None if norm_type is 'none'.

    Raises:
        ValueError: If norm_type is not recognized.
    """
    norm_type = norm_type.lower()
    if norm_type == "none":
        return None
    elif norm_type == "batch":
        return nn.BatchNorm2d(num_features)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_features)
    else:
        raise ValueError(
            f"Unsupported norm type: {norm_type}. "
            f"Choose from 'none', 'batch', 'instance'."
        )


# ============================================================================
# Concrete CNN Implementations
# ============================================================================


class CNNEncoder(BaseEncoder):
    """
    Flexible CNN-based Semantic Encoder.

    Builds a convolutional encoder based on user-specified hidden dimensions,
    kernel sizes, and strides. The output is reshaped to (B, 2, k, H', W')
    following the I/Q convention.

    Args:
        input_channels: Number of input image channels. Default: 3.
        hidden_dims: List of hidden layer dimensions. Length determines
            the number of intermediate conv layers. E.g., [16, 32, 32, 32].
        out_symbol_channels: Number of complex symbols k per spatial location.
            The final conv outputs 2*k channels (I and Q).
        kernel_size: List of kernel sizes for each conv layer (including output layer).
            Length must be len(hidden_dims) + 1.
        stride: List of strides for each conv layer (including output layer).
            Length must be len(hidden_dims) + 1.
        act_type: Activation type ('relu', 'prelu', 'leaky_relu'). Default: 'prelu'.
        norm_type: Normalization type ('none', 'batch', 'instance'). Default: 'none'.

    Example:
        >>> encoder = CNNEncoder(
        ...     input_channels=3,
        ...     hidden_dims=[16, 32, 32, 32],
        ...     out_symbol_channels=8,
        ...     kernel_size=[5, 5, 5, 5, 5],
        ...     stride=[2, 2, 1, 1, 1],
        ...     act_type='prelu',
        ...     norm_type='none'
        ... )
        >>> x = torch.randn(4, 3, 32, 32)
        >>> out = encoder(x)
        >>> out.shape  # (4, 2, 8, 8, 8)
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dims: List[int] = None,
        out_symbol_channels: int = 16,
        kernel_size: List[int] = None,
        stride: List[int] = None,
        act_type: str = "prelu",
        norm_type: str = "none",
    ) -> None:
        """
        Initialize CNNEncoder.

        Args:
            input_channels: Number of input image channels. Default: 3.
            hidden_dims: List of hidden layer dimensions. Default: [16, 32, 32, 32].
            out_symbol_channels: Number of complex symbols k. Default: 16.
            kernel_size: List of kernel sizes. Default: [5, 5, 5, 5, 5].
            stride: List of strides. Default: [2, 2, 1, 1, 1].
            act_type: Activation type. Default: 'prelu'.
            norm_type: Normalization type. Default: 'none'.
        """
        super(CNNEncoder, self).__init__()

        # Default configurations
        if hidden_dims is None:
            hidden_dims = [16, 32, 32, 32]
        if kernel_size is None:
            kernel_size = [5] * (len(hidden_dims) + 1)
        if stride is None:
            stride = [2, 2] + [1] * (len(hidden_dims) - 1)

        # Validate list lengths
        num_layers = len(hidden_dims) + 1  # hidden layers + output layer
        if len(kernel_size) != num_layers:
            raise ValueError(
                f"kernel_size length ({len(kernel_size)}) must equal "
                f"len(hidden_dims) + 1 ({num_layers})."
            )
        if len(stride) != num_layers:
            raise ValueError(
                f"stride length ({len(stride)}) must equal "
                f"len(hidden_dims) + 1 ({num_layers})."
            )

        self.out_symbol_channels = out_symbol_channels

        # Build encoder layers
        layers = []
        in_channels = input_channels

        # Hidden conv layers: Conv -> [Norm] -> Act
        for i, out_channels in enumerate(hidden_dims):
            ks = kernel_size[i]
            s = stride[i]
            padding = ks // 2  # 'same' padding approximation

            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=ks,
                    stride=s,
                    padding=padding,
                )
            )

            # Add normalization if not 'none'
            norm = _get_norm_layer(norm_type, out_channels)
            if norm is not None:
                layers.append(norm)

            # Add activation
            layers.append(_get_activation(act_type))

            in_channels = out_channels

        # Output layer: Conv only (no norm, no activation)
        # Output channels = 2 * k for I/Q components
        out_ks = kernel_size[-1]
        out_s = stride[-1]
        out_padding = out_ks // 2

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=2 * out_symbol_channels,
                kernel_size=out_ks,
                stride=out_s,
                padding=out_padding,
            )
        )

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode input image to semantic symbols.

        Args:
            x: Input image tensor of shape (B, C, H, W).
            **kwargs: Reserved for future extensions.

        Returns:
            Semantic symbols of shape (B, 2, k, H', W'), where:
                - dim=1 index 0 -> In-phase (I) component
                - dim=1 index 1 -> Quadrature (Q) component
                - k = out_symbol_channels

        Shape:
            - Input: (B, C, H, W)
            - Output: (B, 2, k, H', W')
        """
        # (B, C, H, W) -> (B, 2*k, H', W')
        z = self.encoder(x)

        B, _, H_out, W_out = z.shape
        # Reshape: (B, 2*k, H', W') -> (B, 2, k, H', W')
        z = z.view(B, 2, self.out_symbol_channels, H_out, W_out)

        return z


class CNNDecoder(BaseDecoder):
    """
    Flexible CNN-based Semantic Decoder.

    Builds a transposed convolutional decoder (mirror of encoder) based on
    user-specified hidden dimensions. Takes input of shape (B, 2, k, H', W')
    and reconstructs the image.

    Args:
        out_channels: Number of output image channels. Default: 3.
        hidden_dims: List of hidden layer dimensions (should be reverse of
            encoder's hidden_dims for mirrored structure). E.g., [32, 32, 32, 16].
        in_symbol_channels: Number of complex symbols k per spatial location
            (should match encoder's out_symbol_channels).
        kernel_size: List of kernel sizes for each conv transpose layer
            (including output layer). Length must be len(hidden_dims) + 1.
        stride: List of strides for each conv transpose layer (including output layer).
            Length must be len(hidden_dims) + 1.
        act_type: Activation type ('relu', 'prelu', 'leaky_relu'). Default: 'prelu'.
        norm_type: Normalization type ('none', 'batch', 'instance'). Default: 'none'.

    Example:
        >>> decoder = CNNDecoder(
        ...     out_channels=3,
        ...     hidden_dims=[32, 32, 32, 16],
        ...     in_symbol_channels=8,
        ...     kernel_size=[5, 5, 5, 5, 5],
        ...     stride=[1, 1, 1, 2, 2],
        ...     act_type='prelu',
        ...     norm_type='none'
        ... )
        >>> x = torch.randn(4, 2, 8, 8, 8)  # (B, 2, k, H', W')
        >>> out = decoder(x)
        >>> out.shape  # (4, 3, 32, 32)
    """

    def __init__(
        self,
        out_channels: int = 3,
        hidden_dims: List[int] = None,
        in_symbol_channels: int = 16,
        kernel_size: List[int] = None,
        stride: List[int] = None,
        act_type: str = "prelu",
        norm_type: str = "none",
    ) -> None:
        """
        Initialize CNNDecoder.

        Args:
            out_channels: Number of output image channels. Default: 3.
            hidden_dims: List of hidden layer dimensions. Default: [32, 32, 32, 16].
            in_symbol_channels: Number of complex symbols k. Default: 16.
            kernel_size: List of kernel sizes. Default: [5, 5, 5, 5, 5].
            stride: List of strides. Default: [1, 1, 1, 2, 2].
            act_type: Activation type. Default: 'prelu'.
            norm_type: Normalization type. Default: 'none'.
        """
        super(CNNDecoder, self).__init__()

        # Default configurations (mirror of encoder defaults)
        if hidden_dims is None:
            hidden_dims = [32, 32, 32, 16]
        if kernel_size is None:
            kernel_size = [5] * (len(hidden_dims) + 1)
        if stride is None:
            stride = [1] * (len(hidden_dims) - 1) + [2, 2]

        # Validate list lengths
        num_layers = len(hidden_dims) + 1  # hidden layers + output layer
        if len(kernel_size) != num_layers:
            raise ValueError(
                f"kernel_size length ({len(kernel_size)}) must equal "
                f"len(hidden_dims) + 1 ({num_layers})."
            )
        if len(stride) != num_layers:
            raise ValueError(
                f"stride length ({len(stride)}) must equal "
                f"len(hidden_dims) + 1 ({num_layers})."
            )

        self.in_symbol_channels = in_symbol_channels

        # Build decoder layers
        layers = []
        in_ch = 2 * in_symbol_channels  # First layer takes flattened I/Q: (B, 2*k, H', W')

        # Hidden conv transpose layers: ConvT -> [Norm] -> Act
        for i, out_ch in enumerate(hidden_dims):
            ks = kernel_size[i]
            s = stride[i]
            padding = ks // 2
            # output_padding for upsampling with stride > 1
            output_padding = s - 1 if s > 1 else 0

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=ks,
                    stride=s,
                    padding=padding,
                    output_padding=output_padding,
                )
            )

            # Add normalization if not 'none'
            norm = _get_norm_layer(norm_type, out_ch)
            if norm is not None:
                layers.append(norm)

            # Add activation
            layers.append(_get_activation(act_type))

            in_ch = out_ch

        # Output layer: ConvTranspose -> Sigmoid (no norm, no activation before sigmoid)
        out_ks = kernel_size[-1]
        out_s = stride[-1]
        out_padding = out_ks // 2
        out_output_padding = out_s - 1 if out_s > 1 else 0

        layers.append(
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_channels,
                kernel_size=out_ks,
                stride=out_s,
                padding=out_padding,
                output_padding=out_output_padding,
            )
        )
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode received symbols into reconstructed image.

        Args:
            x: Received symbol tensor of shape (B, 2, k, H', W').
            **kwargs: Reserved for future extensions (e.g., CSI).

        Returns:
            Reconstructed image tensor of shape (B, C, H, W),
            with pixel values in [0, 1].

        Shape:
            - Input: (B, 2, k, H', W')
            - Output: (B, C, H, W)
        """
        B, _, k, H_in, W_in = x.shape
        # Flatten I/Q and symbols: (B, 2, k, H', W') -> (B, 2*k, H', W')
        x = x.view(B, 2 * k, H_in, W_in)

        # (B, 2*k, H', W') -> (B, C, H, W)
        out = self.decoder(x)

        return out


# ============================================================================
# Configuration Helper Functions
# ============================================================================


def build_deepjscc_config(cbr: float) -> Dict[str, Any]:
    """
    Build standard DeepJSCC configuration dictionary.

    Creates encoder and decoder configurations following the DeepJSCC paper:
    - 5 conv layers total (4 hidden + 1 output)
    - Hidden dims: [16, 32, 32, 32]
    - Kernel size: 5 for all layers
    - Stride: 2 for first two layers, 1 for remaining layers
    - No BatchNorm (norm_type='none')
    - PReLU activation

    The Channel Bandwidth Ratio (CBR) determines out_symbol_channels (k).
    For an image of size (C, H, W), CBR = (k * H' * W') / (C * H * W).

    For 32x32 RGB images with stride [2,2,1,1,1]:
        - Spatial dims after encoder: H'=W'=8 (32/4=8)
        - CBR = (k * 8 * 8) / (3 * 32 * 32) = 64k / 3072 = k / 48
        - So k = CBR * 48

    Args:
        cbr: Channel Bandwidth Ratio. Typical values: 1/6, 1/12, 1/24, etc.
            This represents the ratio of transmitted complex symbols to
            source image dimensions.

    Returns:
        Dict with keys:
            - 'encoder': Dict of CNNEncoder constructor kwargs.
            - 'decoder': Dict of CNNDecoder constructor kwargs.

    Example:
        >>> config = build_deepjscc_config(cbr=1/6)
        >>> encoder = CNNEncoder(**config['encoder'])
        >>> decoder = CNNDecoder(**config['decoder'])
    """
    # DeepJSCC standard hidden dimensions
    hidden_dims = [16, 32, 32, 32]

    # Kernel size: 5 for all layers (4 hidden + 1 output = 5 layers)
    kernel_size = [5, 5, 5, 5, 5]

    # Stride: first two layers have stride 2 (downsampling), rest have stride 1
    stride = [2, 2, 1, 1, 1]

    # Calculate out_symbol_channels (k) from CBR
    # For 32x32 RGB: k = CBR * 48
    k = max(1, int(cbr * 48))

    # Reverse hidden_dims for decoder (mirror structure)
    decoder_hidden_dims = hidden_dims[::-1]
    decoder_kernel_size = kernel_size[::-1]
    decoder_stride = stride[::-1]

    config = {
        "encoder": {
            "input_channels": 3,
            "hidden_dims": hidden_dims,
            "out_symbol_channels": k,
            "kernel_size": kernel_size,
            "stride": stride,
            "act_type": "prelu",
            "norm_type": "none",
        },
        "decoder": {
            "out_channels": 3,
            "hidden_dims": decoder_hidden_dims,
            "in_symbol_channels": k,
            "kernel_size": decoder_kernel_size,
            "stride": decoder_stride,
            "act_type": "prelu",
            "norm_type": "none",
        },
    }

    return config
