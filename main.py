
import numpy as np
from scripts.data_loading import extract_zip, load_images
from scripts.image_noising import apply_noise_image
from scripts.score import score_autoencoder
from scripts.train import build_autoencoder
from scripts.metrics import ssim_metric, psnr_metric
from scripts.visualization import show_images, show_autoencoder_output, plot_loss_and_metrics, plot_ssim_and_psnr

from tensorflow.keras.optimizers import Adam


if __name__ == '__main__':

    # Path to the zipped dataset file 
    dataset_file_path = '/Data/Dataset.zip'

    # Directory where you want to extract the images
    extraction_path = '/Data/'

    # Extract the dataset
    extract_zip(dataset_file_path, extraction_path)

    # Set dataset parameters
    target_size= (128,128)
    batch_size=64

    # Set train and val directories
    train_dir = extraction_path + '/Dataset/train'
    val_dir = extraction_path+ '/Dataset/validation'

    # Load train and val images
    train_images = load_images(train_dir, target_size)
    val_images = load_images(val_dir, target_size)


    # Apply noise to train and val images
    noised_train_images = np.array([apply_noise_image(image) for image in train_images])
    noised_val_images = np.array([apply_noise_image(image) for image in val_images])

    # Visualize noised images
    show_images(noised_train_images)

    
    # Visualize train images
    show_images(train_images)

    # Define img shape for training
    image_shape = target_size + (3,)

    # Build and train Denoising Autoencoder
    autoencoder = build_autoencoder(image_shape)

    # Compile the model
    # Select the optimizer, loss function and metrics for the model
    learning_rate=0.0001

    optimizer = Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error',metrics=[ssim_metric, psnr_metric])

    # Print model summary
    autoencoder.summary()

    # Set training parameters
    batch_size=32
    n_epochs=1000


    # Train AUtoEncoder
    history = autoencoder.fit(noised_train_images, train_images, batch_size=batch_size, epochs=n_epochs,validation_data=(noised_val_images, val_images) )

    # Evaluate the model
    # Get reconstructed validation images from AutoEncoder
    reconstructed_images = autoencoder.predict(val_images)


    # Compute SSIM and PSNR scores between validation images and reconstructed images from validation images
    average_ssim, average_psnr, ssim_scores, psnr_scores = score_autoencoder(val_images, reconstructed_images)
    print("Average SSIM:", average_ssim)
    print("Average PSNR:", average_psnr)


    # Visualize the outputs of the autoencoder
    show_autoencoder_output(noised_val_images, reconstructed_images, val_images, save_path="Comparisons.png")


    # Plot the training and validation loss curves, SSIM curve, and PSNR curve
    plot_loss_and_metrics(history)

    # Plot SSIM and PSNR values over epochs
    plot_ssim_and_psnr(ssim_scores, psnr_scores)

    # Save trained model as a Keras model
    autoencoder.save('denoising_autoencoder.keras')
    # Save trained model as a TensorFlow SavedModel
    autoencoder.save('denoising_autoencoder.h5')


