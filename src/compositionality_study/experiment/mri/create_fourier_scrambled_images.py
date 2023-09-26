"""Create Fourier-scrambled versions of some images in the selected stimuli directory."""
from typing import Tuple

import cv2
import numpy as np


def fft_phase_scrambling_channel(
    channel: np.ndarray,
    random_seed: int = 42,
) -> np.ndarray:
    """Apply FFT phase scrambling to one channel of an image.

    :param channel: The channel to scramble.
    :type channel: np.ndarray
    :param random_seed: The random seed to use for reproducibility.
    :type random_seed: int
    :return: The scrambled channel.
    :rtype: np.ndarray
    """
    # Step 1: Convert channel to frequency domain
    fft_channel = np.fft.fft2(channel)

    # Step 2: Extract phase and magnitude
    phase_channel = np.angle(fft_channel)
    magnitude_channel = np.abs(fft_channel)

    # Randomize the phase
    np.random.seed(random_seed)  # For reproducibility, change the seed for different results
    phase_offset = 2 * np.pi * np.random.rand(*phase_channel.shape)
    phase_channel = phase_channel + phase_offset

    # Step 3: Combine magnitude and randomized phase to get complex spectrum
    fft_channel_scrambled = magnitude_channel * np.exp(1j * phase_channel)

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

    # Apply phase scrambling for each channel
    b_scrambled = fft_phase_scrambling_channel(b)
    g_scrambled = fft_phase_scrambling_channel(g)
    r_scrambled = fft_phase_scrambling_channel(r)

    # Recombine RGB channels to get the final image
    image_scrambled = cv2.merge((b_scrambled, g_scrambled, r_scrambled))  # noqa

    return image_scrambled
