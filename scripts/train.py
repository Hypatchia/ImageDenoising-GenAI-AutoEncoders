
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU



def build_autoencoder(image_shape):
    """
    Builds a Convolutional AutoEncoder model.

    Parameters:
    - image_shape (tuple): Shape of the input images (height, width, channels).

    Returns:
    - tf.keras.models.Model: Convolutional AutoEncoder model.
    """
    # Weights initialization
    initializer = GlorotNormal()

    # Encoder
    inputs = Input(shape=image_shape)
    x = Conv2D(64, (3, 3), kernel_initializer=initializer, padding='same')(inputs)
    x = LeakyReLU(alpha=0.02)(x)  # Leaky ReLU activation
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.02)(x)  # Leaky ReLU activation
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.02)(x)  # Leaky ReLU activation
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(128, (3, 3), padding='same')(encoded)
    x = LeakyReLU(alpha=0.02)(x)  # Leaky ReLU activation
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.02)(x)  # Leaky ReLU activation
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.02)(x)  # Leaky ReLU activation
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='linear', padding='same')(x)

    # AutoEncoder Model
    autoencoder = tf.keras.models.Model(inputs, decoded)

    return autoencoder

