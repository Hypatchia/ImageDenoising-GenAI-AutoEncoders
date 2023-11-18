

def show_images(images, nmax=4):
    """
    Visualizes a list of images.

    Parameters:
    - images (list): List of images to be visualized.
    - nmax (int, optional): Maximum number of images to display. Defaults to 4.

    Returns:
    - None
    """
    
    # Create subplots based on the number of images to display
    fig, ax = plt.subplots(ncols=min(len(images), nmax), figsize=(12, 4))

    # Iterate through the images and display them
    for i, axi in enumerate(ax.flat):
        axi.imshow(images[i])
        axi.axis('off')

    # Show the plot
    plt.show()



def show_autoencoder_output(noised_val_images, reconstructed_images, val_images, save_path="Comparisons.png"):
    """
    Visualizes the outputs of image processing.

    Parameters:
    - noised_val_images (numpy.ndarray): Noised validation images.
    - reconstructed_images (numpy.ndarray): Reconstructed images.
    - val_images (numpy.ndarray): Original validation images.
    - save_path (str, optional): Path to save the visualization. Defaults to "Comparisons.png".

    Returns:
    - None
    """
    n = len(noised_val_images)
    
    # Increase the figure size
    plt.figure(figsize=(40, 40))

    for i in range(n):
        # Display noisy images
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(noised_val_images[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display denoised images by convolutional autoencoder
        ax = plt.subplot(3, n, i + 1 + n)  # Adjust the index for the second row
        plt.imshow(reconstructed_images[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display original images for comparison purposes
        ax = plt.subplot(3, n, i + 1 + 2*n)  # Adjust the index for the third row
        plt.imshow(val_images[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Reduce spacing between images
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Save the visualization
    plt.savefig(save_path)


def plot_loss_and_metrics(history):
    """
    Plots training and validation loss curves, SSIM curve, and PSNR curve.

    Parameters:
    - history (tf.keras.callbacks.History): Training history of the model.

    Returns:
    - None
    """
    # Create subplots for loss and metrics
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    # Plot the training and validation loss curves
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss Curves')
    axes[0].legend()

    # Plot the SSIM
    axes[1].plot(history.history['ssim_metric'], label='SSIM')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM Curve')
    axes[1].legend()

    # Plot the PSNR
    axes[2].plot(history.history['psnr_metric'], label='PSNR')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('PSNR')
    axes[2].set_title('PSNR Curve')
    axes[2].legend()

    plt.tight_layout()
    plt.show()



def plot_ssim_and_psnr(ssim_scores, psnr_scores):
    """
    Creates a graph to visualize SSIM and PSNR values over epochs.

    Parameters:
    - epochs (range): Range of epochs.
    - ssim_scores (list): List of SSIM scores.
    - psnr_scores (list): List of PSNR scores.

    Returns:
    - None
    """
    epochs = range(len(ssim_scores))
    plt.figure(figsize=(10, 6))

    # Plot SSIM values
    plt.plot(epochs, ssim_scores, label='SSIM', marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()

    # Plot PSNR values
    plt.twinx()
    plt.plot(epochs, psnr_scores, label='PSNR', marker='x', linestyle='--', color='tab:orange')
    plt.ylabel('PSNR')
    plt.legend(loc='upper right')

    plt.title('SSIM and PSNR Comparison')
    plt.show()

