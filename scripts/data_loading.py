
import os 
import zipfile
import numpy as np
import tensorflow as tf
def extract_zip(zip_path, extraction_path):
    """
    Extracts the contents of a zip file to a specified extraction directory.

    Parameters:
    - zip_path (str): Path to the zip file to be extracted.
    - extraction_path (str): Path to the directory where the contents will be extracted.

    Returns:
    - None
    """
    # Create the extraction directory if it doesn't exist
    os.makedirs(extraction_path, exist_ok=True)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)


def is_image_file(filename):
    """
    Checks if a given filename has a valid image extension (JPEG or PNG).

    Parameters:
    - filename (str): The name of the file to be checked.

    Returns:
    bool: True if the file has a valid image extension, False otherwise.
    """

    return filename.lower().endswith((".jpg", ".png"))




def load_images(directory, target_size):
    """
    Loads images from a specified directory, resizes them to a target size, and returns a NumPy array.

    Parameters:
    - directory (str): Path to the directory containing the images.
    - target_size (tuple): A tuple specifying the target size (height, width) to resize the images.

    Returns:
    numpy.ndarray: A NumPy array containing the loaded and resized images.

    """

    images = [np.array(Image.open(os.path.join(directory, filename)).resize(target_size))
              for filename in os.listdir(directory) if is_image_file(filename)]

    # Rescale the images to the range [0, 1]
    images = np.array(images) / 255.0  # Rescale by dividing by 255.0
    images = tf.cast(images, tf.float32)
    return images



