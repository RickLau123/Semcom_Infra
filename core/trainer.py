"""
Training Engine for SemCom_Infra.

This module provides :class:`SemComTrainer`, a general-purpose training
engine for semantic communication systems with:

- **Dynamic SNR training**: uniformly samples SNR from a configurable list
  each batch, forcing the model to generalise across operating points.
- **Per-SNR validation**: iterates over every test SNR independently so
  that performance profiles (PSNR vs. SNR) are clearly visible.
- **TensorBoard logging**: batch-level loss, epoch-level loss / PSNR,
  learning-rate curves.
- **Checkpoint management**: always saves ``last.pth``; saves ``best.pth``
  when validation PSNR improves; saves ``interrupted.pth`` on Ctrl-C.

Example::

    trainer = SemComTrainer(
        system=system,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=torch.optim.Adam(system.parameters(), lr=1e-4),
        scheduler=None,
        device=torch.device('cuda'),
        config={
            'train_snr_list': [1, 4, 7, 10, 13],
            'test_snr_list':  [1, 4, 7, 10, 13, 19],
            'log_dir':        'runs/deepjscc',
            'checkpoint_dir': 'checkpoints/deepjscc',
        },
    )
    trainer.fit(epochs=100)
"""

import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.system import SemComSystem

logger = logging.getLogger(__name__)


class SemComTrainer:
    """
    General-purpose trainer for semantic communication systems.

    Supports dynamic SNR training where the SNR is uniformly sampled
    from a configurable list during training, and per-SNR evaluation
    during validation.

    Args:
        system: End-to-end :class:`SemComSystem` instance.
        train_loader: ``DataLoader`` for the training set.
        val_loader: ``DataLoader`` for the validation set.
        optimizer: PyTorch optimiser (e.g., ``Adam``).
        scheduler: Optional learning-rate scheduler. ``scheduler.step()``
            is called once per epoch after validation.
        device: ``torch.device`` to run on (``'cuda'`` or ``'cpu'``).
        config: Configuration dict — recognised keys:

            ============== ======== =======================================
            Key            Default  Description
            ============== ======== =======================================
            train_snr_list [1,4,7,  SNR values (dB) sampled during training
                           10,13]
            test_snr_list  [1,4,7,  SNR values (dB) evaluated one-by-one
                           10,13,   during validation
                           19]
            log_dir        'runs/'  TensorBoard log directory
            checkpoint_dir 'ckpts/' Checkpoint save directory
            ============== ======== =======================================
    """

    def __init__(
        self,
        system: SemComSystem,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        config: Dict,
    ) -> None:
        self.system = system.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # ----- Config -----
        self.train_snr_list: List[float] = config.get(
            "train_snr_list", [1, 4, 7, 10, 13]
        )
        self.test_snr_list: List[float] = config.get(
            "test_snr_list", [1, 4, 7, 10, 13, 19]
        )
        log_dir = config.get("log_dir", "runs/")
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints/"))

        # ----- Loss -----
        self.mse_loss = nn.MSELoss()

        # ----- TensorBoard -----
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # ----- Checkpoint directory -----
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Global step counter for batch-level TensorBoard logging
        self.global_step: int = 0

    # ================================================================== SNR
    def _sample_snr(self, batch_size: int) -> float:
        """
        Sample a random SNR (dB) uniformly from the training SNR list.

        One SNR value is drawn per batch (not per sample) because the
        underlying channel modules accept a scalar ``snr_db``.  The
        ``batch_size`` parameter is accepted for interface symmetry and
        potential future per-sample SNR extensions.

        Args:
            batch_size: Current batch size (reserved for future use).

        Returns:
            A scalar ``float`` SNR value in dB.
        """
        idx = np.random.randint(0, len(self.train_snr_list))
        return float(self.train_snr_list[idx])

    # ============================================================ train_epoch
    def train_epoch(self, epoch: int) -> float:
        """
        Run one training epoch.

        For each batch a random SNR is sampled from the training list
        and fed through the system.  Batch loss is logged to TensorBoard
        at every step.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Average MSE loss over the epoch.
        """
        self.system.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1} [Train]",
            leave=False,
        )

        for batch in pbar:
            img = batch.to(self.device)  # (B, C, H, W)

            # Sample random SNR for this batch
            snr_db = self._sample_snr(img.size(0))

            # Forward
            out = self.system(img, snr_db)
            loss = self.mse_loss(out, img)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ---- Logging ----
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            self.global_step += 1

            self.writer.add_scalar(
                "Train/BatchLoss", batch_loss, self.global_step
            )
            pbar.set_postfix(loss=f"{batch_loss:.4f}", snr=f"{snr_db:.1f}dB")

        avg_loss = total_loss / max(num_batches, 1)
        self.writer.add_scalar("Train/EpochLoss", avg_loss, epoch)

        return avg_loss

    # ============================================================== validate
    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """
        Validate across all test SNR values.

        Unlike training (which randomly samples one SNR per batch),
        validation iterates over **each** SNR in ``test_snr_list``
        separately and reports per-SNR metrics, providing a complete
        performance profile.

        Metrics logged to TensorBoard:
            - ``Val/PSNR_{snr}dB``  — per-SNR PSNR
            - ``Val/MSE_{snr}dB``   — per-SNR MSE
            - ``Val/PSNR_avg``      — mean PSNR across all test SNRs

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Average PSNR (dB) across all test SNR values.
        """
        self.system.eval()

        psnr_per_snr: Dict[float, float] = {}

        for snr_db in self.test_snr_list:
            total_mse = 0.0
            total_samples = 0

            for batch in self.val_loader:
                img = batch.to(self.device)  # (B, C, H, W)
                out = self.system(img, snr_db)

                # Per-pixel MSE, accumulated by sample count
                mse = torch.mean((out - img) ** 2).item()
                total_mse += mse * img.size(0)
                total_samples += img.size(0)

            avg_mse = total_mse / max(total_samples, 1)

            # PSNR = 10 * log10(1 / MSE),  assuming data ∈ [0, 1]
            psnr = 10.0 * math.log10(1.0 / max(avg_mse, 1e-10))
            psnr_per_snr[snr_db] = psnr

            # Per-SNR TensorBoard tags
            self.writer.add_scalar(f"Val/PSNR_{snr_db}dB", psnr, epoch)
            self.writer.add_scalar(f"Val/MSE_{snr_db}dB", avg_mse, epoch)

        # Mean PSNR across all test SNRs
        avg_psnr = float(np.mean(list(psnr_per_snr.values())))
        self.writer.add_scalar("Val/PSNR_avg", avg_psnr, epoch)

        # Readable per-SNR summary
        snr_str = " | ".join(
            f"{snr:.0f}dB: {psnr:.2f}"
            for snr, psnr in psnr_per_snr.items()
        )
        logger.info(f"[Val] PSNR by SNR: {snr_str}")

        return avg_psnr

    # ======================================================= _save_checkpoint
    def _save_checkpoint(
        self, path: Path, epoch: int, val_psnr: float
    ) -> None:
        """
        Save a training checkpoint.

        The checkpoint contains model weights, optimiser state,
        scheduler state, and metadata needed to resume training.

        Args:
            path: Destination file path.
            epoch: Current epoch number.
            val_psnr: Validation PSNR at this epoch.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.system.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "val_psnr": val_psnr,
                "global_step": self.global_step,
            },
            path,
        )

    # ================================================================== fit
    def fit(self, epochs: int) -> None:
        """
        Main training loop.

        Runs ``train_epoch`` → ``validate`` → checkpoint for each epoch.

        Saving strategy:
            - ``last.pth``  — saved after **every** epoch.
            - ``best.pth``  — saved when ``val_psnr > best_psnr``.
            - ``interrupted.pth`` — saved on ``KeyboardInterrupt`` (Ctrl-C).

        Args:
            epochs: Total number of epochs to train.
        """
        best_psnr = -float("inf")

        logger.info(
            f"Starting training for {epochs} epochs | "
            f"Train SNRs: {self.train_snr_list} | "
            f"Test SNRs: {self.test_snr_list}"
        )

        # Track for safe KeyboardInterrupt handling
        epoch = 0
        val_psnr = 0.0

        try:
            for epoch in range(epochs):
                t0 = time.time()

                # ---- Train ----
                train_loss = self.train_epoch(epoch)

                # ---- Validate ----
                val_psnr = self.validate(epoch)

                # ---- Scheduler step ----
                if self.scheduler is not None:
                    self.scheduler.step()

                elapsed = time.time() - t0

                # ---- Console log ----
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val PSNR: {val_psnr:.2f} dB | "
                    f"LR: {lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
                self.writer.add_scalar("Train/LR", lr, epoch)

                # ---- Save last ----
                self._save_checkpoint(
                    self.checkpoint_dir / "last.pth", epoch, val_psnr
                )

                # ---- Save best ----
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    self._save_checkpoint(
                        self.checkpoint_dir / "best.pth", epoch, val_psnr
                    )
                    print(
                        f"  -> New best PSNR: {best_psnr:.2f} dB  (saved)"
                    )

        except KeyboardInterrupt:
            print("\n[!] Training interrupted by user. Saving checkpoint...")
            self._save_checkpoint(
                self.checkpoint_dir / "interrupted.pth", epoch, val_psnr
            )
            print(
                f"  -> Checkpoint saved to "
                f"{self.checkpoint_dir / 'interrupted.pth'}"
            )

        finally:
            self.writer.close()
            logger.info(
                f"Training finished. Best Val PSNR: {best_psnr:.2f} dB"
            )
