import tensorflow as tf

def ssim_metric(original, reconstructed):
    """
    Compute Structural Similarity Index (SSIM) between two images.

    SSIM is a metric that measures the similarity between two images, taking into account luminance, contrast,
    and structure. It returns a value between -1 and 1, where 1 indicates identical images.

    Parameters:
    - original (tf.Tensor): The original image.
    - reconstructed (tf.Tensor): The reconstructed image.

    Returns:
    - float: SSIM value between the original and reconstructed images.
    """
    ssim = tf.image.ssim(original, reconstructed, max_val=1.0)
    return ssim

def psnr_metric(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR is a metric that measures the quality of the reconstructed image compared to the original image.
    It is expressed in decibels (dB), and higher values indicate better image quality.

    Parameters:
    - img1 (tf.Tensor): The original image.
    - img2 (tf.Tensor): The reconstructed image.

    Returns:
    - float: PSNR value between the original and reconstructed images.
    """
    psnr = tf.image.psnr(img1, img2, max_val=1.0)
    return psnr
