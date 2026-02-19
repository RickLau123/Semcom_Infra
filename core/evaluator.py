"""
Evaluation Engine for SemCom_Infra.

This module provides :class:`SemComEvaluator`, a dedicated testing and
evaluation engine for trained semantic communication systems.

Features:

- **Per-SNR evaluation**: tests the model at each SNR operating point
  independently, producing a complete PSNR-vs-SNR performance profile.
- **CSV export**: saves ``{SNR, PSNR}`` results to a CSV file for
  reproducible record-keeping and downstream analysis.
- **Visualisation**: generates and saves a publication-ready PSNR-vs-SNR
  curve via Matplotlib.

Example::

    evaluator = SemComEvaluator(
        system=system,
        test_loader=test_loader,
        device=torch.device('cuda'),
    )
    evaluator.load_weights('checkpoints/deepjscc/best.pth')
    results = evaluator.run(
        snr_list=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        save_dir='results/deepjscc',
    )
    evaluator.plot_results(
        snr_list=list(results.keys()),
        psnr_list=list(results.values()),
        save_dir='results/deepjscc',
    )
"""

import logging
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.system import SemComSystem

logger = logging.getLogger(__name__)


class SemComEvaluator:
    """
    Evaluation engine for trained semantic communication systems.

    Loads a trained checkpoint, iterates over every requested SNR
    operating point, and reports per-SNR PSNR metrics with optional
    CSV export and curve plotting.

    Args:
        system: End-to-end :class:`SemComSystem` instance.
        test_loader: ``DataLoader`` for the test / evaluation set.
        device: ``torch.device`` to run on (``'cuda'`` or ``'cpu'``).
    """

    def __init__(
        self,
        system: SemComSystem,
        test_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.system = system.to(device)
        self.system.eval()
        self.test_loader = test_loader
        self.device = device

    # ======================================================= load_weights
    def load_weights(self, checkpoint_path: str) -> None:
        """
        Load trained model weights from a ``.pth`` checkpoint file.

        Only the ``model_state_dict`` key is used; optimiser / scheduler
        states are ignored.

        Args:
            checkpoint_path: Path to the ``.pth`` checkpoint file.

        Raises:
            FileNotFoundError: If *checkpoint_path* does not exist.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file not found: '{checkpoint_path}'"
            )

        checkpoint = torch.load(
            ckpt_path, map_location=self.device, weights_only=True
        )
        self.system.load_state_dict(checkpoint["model_state_dict"])
        self.system.eval()

        epoch = checkpoint.get("epoch", "?")
        val_psnr = checkpoint.get("val_psnr", "?")
        logger.info(
            f"Loaded checkpoint from '{checkpoint_path}' "
            f"(epoch={epoch}, val_psnr={val_psnr})"
        )
        print(
            f"[✓] Loaded weights from '{checkpoint_path}' "
            f"(epoch={epoch}, val_psnr={val_psnr})"
        )

    # ====================================================== evaluate_at_snr
    @torch.no_grad()
    def evaluate_at_snr(self, snr_db: float) -> float:
        """
        Evaluate the system at a single SNR operating point.

        Iterates over the entire test set, computes per-batch MSE,
        and returns the average PSNR across all samples.

        Args:
            snr_db: Channel SNR in dB.

        Returns:
            Average PSNR (dB) over the test set at the given SNR.
            Assumes pixel values are normalised to ``[0, 1]``.
        """
        self.system.eval()

        total_mse = 0.0
        total_samples = 0

        for batch in self.test_loader:
            img = batch.to(self.device)  # (B, C, H, W)
            out = self.system(img, snr_db)

            # Per-pixel MSE, accumulated by sample count
            mse = torch.mean((out - img) ** 2).item()
            total_mse += mse * img.size(0)
            total_samples += img.size(0)

        avg_mse = total_mse / max(total_samples, 1)

        # PSNR = 10 * log10(1 / MSE),  assuming data ∈ [0, 1]
        psnr = 10.0 * math.log10(1.0 / max(avg_mse, 1e-10))

        return psnr

    # ================================================================= run
    def run(
        self,
        snr_list: List[float],
        save_dir: str = "results/",
    ) -> Dict[float, float]:
        """
        Evaluate the system across multiple SNR operating points.

        For each SNR in *snr_list*, calls :meth:`evaluate_at_snr` and
        collects the results.  A summary CSV is saved to
        ``save_dir/results.csv``.

        Args:
            snr_list: List of SNR values (dB) to evaluate.
            save_dir: Directory to save the results CSV.

        Returns:
            Dictionary mapping ``{snr_db: psnr_db}`` for every
            evaluated operating point.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        results: Dict[float, float] = {}

        pbar = tqdm(snr_list, desc="Evaluating", leave=True)
        for snr_db in pbar:
            psnr = self.evaluate_at_snr(snr_db)
            results[snr_db] = psnr
            pbar.set_postfix(SNR=f"{snr_db:.1f}dB", PSNR=f"{psnr:.2f}dB")
            print(f"  Testing at SNR={snr_db:.1f} dB: PSNR={psnr:.2f} dB")

        # ---- Save to CSV ----
        df = pd.DataFrame(
            {"SNR_dB": list(results.keys()), "PSNR_dB": list(results.values())}
        )
        csv_path = save_path / "results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to '{csv_path}'")
        print(f"[✓] Results saved to '{csv_path}'")

        # ---- Log summary ----
        snr_str = " | ".join(
            f"{snr:.0f}dB: {psnr:.2f}"
            for snr, psnr in results.items()
        )
        logger.info(f"[Eval] PSNR by SNR: {snr_str}")

        return results

    # =========================================================== plot_results
    def plot_results(
        self,
        snr_list: List[float],
        psnr_list: List[float],
        save_dir: str = "results/",
    ) -> None:
        """
        Plot and save a PSNR-vs-SNR performance curve.

        Generates a Matplotlib figure with grid lines, axis labels,
        legend, and a descriptive title.  The figure is saved to
        ``save_dir/psnr_curve.png``.

        Args:
            snr_list: List of SNR values (dB) — x-axis.
            psnr_list: Corresponding PSNR values (dB) — y-axis.
            save_dir: Directory to save the plot image.
        """
        # Deferred import so that matplotlib is optional at module level
        import matplotlib.pyplot as plt

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            snr_list,
            psnr_list,
            marker="o",
            linewidth=2,
            markersize=6,
            label="DeepJSCC",
        )
        ax.set_xlabel("SNR (dB)", fontsize=13)
        ax.set_ylabel("PSNR (dB)", fontsize=13)
        ax.set_title("DeepJSCC Performance", fontsize=15)
        ax.legend(fontsize=12)
        ax.grid(True)

        fig_path = save_path / "psnr_curve.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"PSNR curve saved to '{fig_path}'")
        print(f"[✓] PSNR curve saved to '{fig_path}'")
