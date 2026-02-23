# SemCom_Infra Core Components

`SemCom_Infra` is a PyTorch-based semantic communication research infrastructure library designed to provide modular components for simulating and training end-to-end communication systems. The core components are located in the `core` folder, and this document provides an overview of these components.

---

## Core Modules Overview

### 1. **`core.channel`**
- **Functionality**: Provides differentiable channel models, including:
  - **AWGNChannel**: Additive White Gaussian Noise channel.
  - **RayleighChannel**: Rayleigh fading channel supporting complex signal multiplication and noise addition.
- **Features**:
  - Dynamic adjustment of channel noise (based on SNR).
  - Differentiable for backpropagation, suitable for end-to-end training.

---

### 2. **`core.codec`**
- **Functionality**: Defines semantic encoder and decoder base classes and implementations:
  - **BaseEncoder**: Abstract semantic encoder for encoding images into semantic symbols.
  - **BaseDecoder**: Abstract semantic decoder for recovering images from semantic symbols.
  - **CNNEncoder**: Flexible encoder implementation based on CNNs.
  - **CNNDecoder**: Flexible decoder implementation based on CNNs.
- **Features**:
  - Customizable hidden layer structures, kernel sizes, and strides.
  - Outputs symbols in the format `(B, 2, k, H', W')` for complex signals.

---

### 3. **`core.dataset`**
- **Functionality**: Provides data loading and preprocessing tools:
  - **ImageDataset**: Supports loading image data from local folders or Hugging Face Hub.
  - **get_standard_transforms**: Provides standard data augmentation and transformation pipelines.
  - **get_dataloader**: Quickly builds PyTorch `DataLoader`.
- **Features**:
  - Different data augmentation strategies for training and validation modes.
  - Compatible with various image formats (JPEG, PNG, BMP, etc.).

---

### 4. **`core.evaluator`**
- **Functionality**: Provides model evaluation tools:
  - **SemComEvaluator**: Tests trained semantic communication systems and generates PSNR-SNR performance curves.
- **Features**:
  - Supports performance evaluation at individual SNR points.
  - Automatically generates CSV files and performance plots for analysis and visualization.

---

### 5. **`core.modem`**
- **Functionality**: Defines modulation and demodulation modules (Modem), including:
  - **BaseModem**: Abstract base class supporting power normalization and complex signal conversion.
  - **AnalogModem**: Analog modem supporting end-to-end semantic communication.
- **Features**:
  - Power normalization ensures $E[x^2] = 1$.
  - Supports conversion between complex signals and real-valued tensors.

---

### 6. **`core.system`**
- **Functionality**: Defines end-to-end semantic communication systems:
  - **SemComSystem**: Integrates encoder, modem, channel, and decoder into a differentiable `nn.Module`.
- **Features**:
  - Supports the complete semantic communication data flow: `Image -> Encoder -> Modem -> Channel -> Decoder -> Image_hat`.
  - Provides flexible extension interfaces for custom modules.

---

### 7. **`core.trainer`**
- **Functionality**: Provides a training engine:
  - **SemComTrainer**: Supports dynamic SNR training, per-SNR validation, TensorBoard logging, and model checkpoint management.
- **Features**:
  - Dynamic sampling of training SNRs to enhance model generalization.
  - Automatically saves the best model weights and intermediate checkpoints.

---

### 8. **`core.utils`**
- **Functionality**: Provides utility functions:
  - **tensor_to_image**: Converts normalized tensors to images with pixel values in the range `[0, 255]`.
- **Features**:
  - Simplifies tensor-to-image conversion for visualization and saving.

---

## Installation and Usage

### Installation
1. Ensure Python 3.8+ and PyTorch are installed.
2. Clone this repository:
   ```bash
   git clone https://github.com/your_username/SemCom_Infra.git
   cd SemCom_Infra
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Example Usage
Below is a simple end-to-end training example:
```python
from core.codec import CNNEncoder, CNNDecoder
from core.modem import AnalogModem
from core.channel import AWGNChannel
from core.system import SemComSystem
from core.trainer import SemComTrainer
from core.dataset import ImageDataset, get_standard_transforms, get_dataloader

# Data loading
transform = get_standard_transforms(crop_size=128, is_train=True)
dataset = ImageDataset(source="./data/UIEB/train", source_type="local", transform=transform)
train_loader = get_dataloader(dataset, batch_size=16, shuffle=True)

# Model initialization
encoder = CNNEncoder(hidden_dims=[16, 32, 32], out_symbol_channels=32)
decoder = CNNDecoder(hidden_dims=[32, 32, 16], in_symbol_channels=32)
modem = AnalogModem()
channel = AWGNChannel()
system = SemComSystem(encoder, decoder, modem, channel)

# Training
trainer = SemComTrainer(
    system=system,
    train_loader=train_loader,
    val_loader=None,
    optimizer=torch.optim.Adam(system.parameters(), lr=1e-4),
    scheduler=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    config={"train_snr_list": [1, 4, 7], "log_dir": "runs/", "checkpoint_dir": "checkpoints/"},
)
trainer.fit(epochs=10)
```

---

## Contribution Guidelines
We welcome contributions! Please follow these steps:
1. Fork this repository.
2. Create a new branch and commit your changes.
3. Submit a Pull Request.

---

## License
This project is open-sourced under the [MIT License](LICENSE).