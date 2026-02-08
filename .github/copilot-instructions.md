# Copilot Instructions for SemCom_Infra

## Build, Test, and Lint Commands
- This repository is based on PyTorch. If you add tests, use `pytest` or `unittest` conventions.
- To run a single test (if present): `pytest path/to/test_file.py -k test_function_name` or `python -m unittest path.to.test_file.TestClass.test_method`.
- No explicit build or lint commands are defined yet; follow standard Python practices (e.g., `flake8`, `black`).

## High-Level Architecture
- **Source/Destination:** Image tensors in shape `(Batch, Channel, Height, Width)`.
- **Encoder (Tx):** Semantic encoder outputs feature maps or semantic vectors.
- **Modem:** Handles complex <-> real conversion, power normalization (ensure $E[x^2]=1$), and physical layer modulation (e.g., OFDM). All OFDM logic belongs here.
- **Channel:** Must be differentiable for backpropagation. Supports models like AWGN, Rayleigh, Fading. Input/output are real-valued signals from/to Modem.
- **Decoder (Rx):** Recovers images from noisy features.
- **Directory Structure:**
  - `core/`: Abstract base classes (`BaseEncoder`, `BaseDecoder`, `BaseChannel`, `BaseModem`).
  - `models/`: Paper implementations (e.g., `DeepJSCC`, `WITT`).
  - `utils/`: Metrics (`PSNR`, `SSIM`) and helpers.

## Key Conventions
- **Tensor Shapes:**
  - Image input: Always `(Batch_Size, Channel, Height, Width)`.
  - Complex signals: Use real tensors with shape `(B, 2, ...)`, where `dim=1` is [I, Q].
- **Physical Layer:**
  - SNR input is dB; convert to linear: `SNR_linear = 10 ** (SNR_dB / 10)`.
  - Power normalization: `x_norm = x / sqrt(E[x^2] + epsilon)`.
- **PyTorch:**
  - All modules inherit from `nn.Module`.
  - Never use `torch.no_grad()` in training `forward`.
  - Use type hints in all function signatures.
  - Prefer `einops` for complex reshaping.
- **Other:**
  - OFDM logic is in Modem, not Channel or Encoder.
  - Avoid hardcoding values (e.g., `image_size=32`, `snr=10`); pass via `__init__`.
  - For reshape/view, comment tensor shape changes to avoid silent errors.

---

If you add tests, configs, or new conventions, update this file to help Copilot and future contributors.
