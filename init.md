# SemCom_Infra 项目指南

## 项目概述

这是一个基于 PyTorch 构建的**语义通信 (Semantic Communication) 科研基础设施库(Infra)**。

也就是 SemCom 领域的 "Hugging Face Transformers"。

核心目标是提供开箱即用的模块（Encoder, Decoder, Channel, Modem），支持从图像信源（Image Source）开始的端到端（End-to-End）通信系统仿真与训练。

## 核心架构设计 (Architecture)

- **Source/Destination:** 图像张量，标准格式 `(B, C, H, W)`。
- **Encoder (Tx):** 语义编码器，输出特征图或语义向量。
- **Modem (调制解调模块):** **关键模块**。
  - 负责 **复数 <-> 实数转换**。
  - 负责 **功率归一化 (Power Normalization)**，确保进入信道前 $E[x^2]=1$。
  - 负责 OFDM 等物理层波形调制。
- **Channel (信道模块):**
  - **必须可微 (Differentiable)** 以支持反向传播。
  - 输入/输出均为经过 Modem 处理后的物理信号。
  - 支持 `AWGN`, `Rayleigh`, `Fading` 等信道模型。
- **Decoder (Rx):** 语义解码器，从带噪特征中恢复图像。

## 代码规范与约束 (Code Style & Constraints)

- **张量形状 (Tensor Shapes):**
  - **图像输入:** 始终保持 `(Batch_Size, Channel, Height, Width)`。
  - **复数信号:** 使用 **实数张量** 模拟。约定形状为 `(B, 2, ...)`，其中 `dim=1` 的两个通道分别代表实部 (I) 和虚部 (Q)。
- **物理层约束 (Physics):**
  - **SNR 处理:** 用户输入通常为 **dB**。计算噪声功率时必须先转换为线性值：$SNR_{linear} = 10^{(SNR_{dB}/10)}$。
  - **功率归一化:** 发送端信号必须归一化。公式：$x_{norm} = x / \sqrt{E[x^2] + \epsilon}$。
- **PyTorch 规范:**
  - 所有模块必须继承自 `nn.Module`。
  - **严禁** 在训练路径的 `forward` 函数中使用 `torch.no_grad()`。
  - 必须使用 Type Hints (e.g., `def forward(self, x: torch.Tensor) -> torch.Tensor:`)。
  - 优先使用 `einops` 进行复杂的维度变换（rearrange/reduce）。

## 目录结构 (Directory Structure)

- `core/`: 存放抽象基类 (`BaseEncoder`, `BaseDecoder`, `BaseChannel`, `BaseModem`)。
- `models/`: 存放具体论文复现 (e.g., `DeepJSCC`, `WITT`)。
- `utils/`: 存放评估指标 (`PSNR`, `SSIM`) 和辅助工具。

## 注意事项 (Critical Notes)

- **OFDM 归属:** OFDM 调制/解调逻辑属于 `Modem` 模块，**不**属于 Channel 或 Encoder。
- **避免硬编码:** 不要将 `image_size=32` 或 `snr=10` 写死在类定义中，必须通过 `__init__` 参数传入。
- **维度陷阱:** 在进行 `Reshape` 或 `View` 操作前，务必在注释中推导张量形状变化，防止静默的维度错误。