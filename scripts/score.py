import numpy as np
import tensorflow as tf
from metrics import ssim_metric, psnr_metric



def score_autoencoder(val_images, reconstructed_images):
    """
    Calculates SSIM and PSNR scores between validation images and reconstructed images.

    Parameters:
    - val_images (numpy.ndarray): Validation images.
    - reconstructed_images (numpy.ndarray): Reconstructed images.

    Returns:
    - float: Average SSIM (Structural Similarity Index).
    - float: Average PSNR (Peak Signal-to-Noise Ratio).
    """
    ssim_scores = []
    psnr_scores = []

    # Compute SSIM and PSNR between val images and reconstructed validation images
    for i in range(len(val_images)):
        ssim_score = ssim_metric(val_images[i], reconstructed_images[i])
        psnr_score = psnr_metric(val_images[i], reconstructed_images[i])
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)

    # Calculate average SSIM and PSNR
    average_ssim = np.mean(ssim_scores)
    average_psnr = np.mean(psnr_scores)

    return average_ssim, average_psnr, ssim_scores, psnr_scores



