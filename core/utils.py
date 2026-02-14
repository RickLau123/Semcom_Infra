import torch

def tensor_to_image(tensor: torch.Tensor):
    """
    Convert a normalized tensor to an image with pixel values in [0, 255].

    This function takes a tensor with pixel values normalized to [0, 1] (e.g.,
    the output of a decoder with a Sigmoid activation) and scales it to the
    range [0, 255] for visualization or saving as an image.

    Args:
        tensor: Input tensor with pixel values in [0, 1].

    Returns:
        torch.Tensor: Tensor with pixel values in [0, 255], rounded to integers.

    Example:
        >>> tensor = torch.rand(1, 3, 32, 32)  # Normalized tensor
        >>> image = tensor_to_image(tensor)
        >>> image.min(), image.max()  # (0, 255)
    """
    return (tensor.clamp(0, 1) * 255.0).round()