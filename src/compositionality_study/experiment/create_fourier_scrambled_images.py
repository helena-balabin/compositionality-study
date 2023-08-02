"""Create Fourier-scrambled versions of some images in the selected stimuli directory."""
import cv2
import numpy as np


def fft_phase_scrambling_channel(
    channel: np.ndarray,
    avg_phase: np.ndarray,
    random_seed: int = 42,
) -> np.ndarray:
    """Apply FFT phase scrambling to one channel of an image, taking the avg phase spectrum of the image into account.

    :param channel: The channel to scramble.
    :type channel: np.ndarray
    :param avg_phase: The average phase spectrum of the image.
    :type avg_phase: np.ndarray
    :param random_seed: The random seed to use for reproducibility.
    :type random_seed: int
    :return: The scrambled channel.
    :rtype: np.ndarray
    """
    # Step 1: Convert channel to frequency domain
    fft_channel = np.fft.fft2(channel)

    # Step 2: Randomize phase in frequency domain using the average phase as reference
    phase_channel = np.angle(fft_channel)

    # Apply a random phase offset to each pixel in the frequency domain
    np.random.seed(random_seed)  # For reproducibility, change the seed for different results
    phase_offset = 2 * np.pi * np.random.rand(*phase_channel.shape)
    phase_channel = np.angle(np.exp(1j * (phase_channel - avg_phase) + 1j * phase_offset))

    # Step 3: Combine magnitude and randomized phase to get complex spectrum
    mag_channel = np.abs(fft_channel)
    fft_channel_scrambled = mag_channel * np.exp(1j * phase_channel)

    # Step 4: Inverse FFT to get scrambled channel
    channel_scrambled = np.abs(np.fft.ifft2(fft_channel_scrambled)).astype(np.uint8)

    # Return the scrambled channel
    return channel_scrambled


def fft_phase_scrambling(
    im_path: str,
) -> np.ndarray:
    """Apply Fourier phase scrambling to an (RGB) image.

    :param im_path: The path to the image.
    :type im_path: str
    :return: The scrambled image.
    :rtype: np.ndarray
    """
    # Step 1: Load image
    image = cv2.imread(im_path)  # noqa

    # Separate RGB channels
    b, g, r = cv2.split(image)  # noqa

    # Calculate the average phase spectrum using the blue channel
    # It doesn't matter which channel we use, because the phase relationships among the channels in an RGB image are
    # consistent
    avg_phase = np.angle(np.fft.fft2(b))

    # Apply phase scrambling for each channel with the average phase as reference
    b_scrambled = fft_phase_scrambling_channel(b, avg_phase)
    g_scrambled = fft_phase_scrambling_channel(g, avg_phase)
    r_scrambled = fft_phase_scrambling_channel(r, avg_phase)

    # Recombine RGB channels to get the final image
    image_scrambled = cv2.merge((b_scrambled, g_scrambled, r_scrambled))  # noqa

    return image_scrambled
