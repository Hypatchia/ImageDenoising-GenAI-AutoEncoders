


import numpy as np
import tensorflow as tf

def apply_noise_image(image, scale=10):
    """
    Applies Gaussian noise to an input image.

    Parameters:
    - image (numpy.ndarray): Input image.
    - scale (float, optional): Scaling factor for the noise intensity. Defaults to 10.

    Returns:
    - numpy.ndarray: Noised image.
    """

    # Generate Gaussian noise with the same shape as the image
    noise = np.random.randn(*image.shape)

    # Scale the noise to fit the [0, 1] range
    noise = noise * (scale / 255.0)  # Assuming the original image was in the range [0, 255]

    # Add the scaled noise to the image
    noised_image = image + noise

    # Ensure that the resulting image is still within [0, 1]
    noised_image = np.clip(noised_image, 0, 1)
    
    # Convert the image to float32
    noised_image = tf.cast(noised_image, tf.float32)

    return noised_image