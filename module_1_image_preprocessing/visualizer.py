from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .models import PreprocessResult


def visualize_result(
    original: np.ndarray,
    result: PreprocessResult,
    save_path: str | Path | None = None,
) -> None:
    """
    Render a 1x2 debug figure:
      Left  — original image with barcode bounding boxes overlaid
      Right — processed image (or error message on failure)

    Metadata strip at the bottom shows blank/orientation/angle/barcode/status.
    Pass save_path to write to disk; omit to call plt.show() interactively.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Module 1 — Image Preprocessing Result", fontsize=13, fontweight="bold")

    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for bc in result.barcodes:
        bb = bc.bbox
        axes[0].add_patch(patches.Rectangle(
            (bb["x"], bb["y"]), bb["width"], bb["height"],
            linewidth=2, edgecolor="lime", facecolor="none",
        ))
        axes[0].text(
            bb["x"], bb["y"] - 6, bc.barcode_type,
            color="lime", fontsize=8, fontweight="bold",
        )

    if result.processed_image is not None:
        # Tự động nhận diện: Nếu ảnh có 3 kênh (ảnh màu BGR) -> Phải chuyển sang RGB
        if len(result.processed_image.shape) == 3:
            proc_rgb = cv2.cvtColor(result.processed_image, cv2.COLOR_BGR2RGB)
            axes[1].imshow(proc_rgb)
        else:
            # Nếu ảnh có 2 kênh (Trắng đen / Grayscale) -> Dùng cmap gray
            axes[1].imshow(result.processed_image, cmap="gray")
        axes[1].set_title("Processed (crop → deskew → enhance)")

    barcode_str = (
        f"{len(result.barcodes)} ({result.barcodes[0].barcode_type})"
        if result.barcodes else "0"
    )
    status_str = "OK" if result.error_code is None else result.error_code
    info = (
        f"Blank: {result.is_blank}  |  "
        f"Orientation: {result.orientation_status.value}  |  "
        f"Skew: {result.skew_angle:.1f}°  |  "
        f"Barcodes: {barcode_str}  |  "
        f"Warnings: {', '.join(result.warnings) or 'none'}  |  "
        f"Status: {status_str}"
    )
    fig.text(0.02, 0.01, info, fontsize=8.5, color="#333333")

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
