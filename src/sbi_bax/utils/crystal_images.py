import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sbi_bax.calculate.crystal import count_bright_peaks


def visualize_image(image: np.ndarray, angle: torch.Tensor, fig_name: str):
    """
    Visualize a crystal diffraction image.

    Args:
        image (np.ndarray): The diffraction image to visualize.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image, vmin=0, vmax=3)
    plt.title(
        f"Angle: {angle.cpu().numpy()} \nBright Peaks: {count_bright_peaks(image)}"
    )
    plt.colorbar(label="Intensity")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.savefig(fig_name)
    plt.close()


def visualize_images(images: np.ndarray, angles: np.ndarray, output_dir: Path):
    """
    Visualize a set of crystal diffraction images.

    Args:
        images (np.ndarray): Array of diffraction images to visualize.
        output_dir (Path): Directory to save the visualizations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        visualize_image(image, angles[i], output_dir / f"image_{i}.png")
