"""
TODO: add descriptions
"""

import os
import cv2
import numpy as np
from PIL import Image

# Handling DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = 200000000 # TODO: check


def get_image_size(image_path):
    """
    Function to get the size of an image.

    :param image_path: Path to the image file.
    :return: Tuple containing width and height of the image.
    """
    with Image.open(image_path) as img:
        return img.size


def find_large_images(directory, max_pixels):
    """
    Function to find images in a directory that exceed a certain number of pixels.

    :param directory: Path to the directory containing images.
    :param max_pixels: Maximum number of pixels (width * height).
    :return: List of image paths that exceed the maximum number of pixels.
    """
    large_images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            image_path = os.path.join(directory, filename)
            width, height = get_image_size(image_path)
            if width * height > max_pixels:
                large_images.append(image_path)
    return large_images


def check_image_integrity(processed_image, original_image_path):
    """
    Check the integrity of a processed image against the original image.

    :param processed_image: The processed image to check, as a NumPy array.
    :param original_image_path: Path to the original image for comparison.
    :return: True if the image passes all integrity checks, False otherwise.
    """
    try:
        # Load the original image
        original_img = cv2.imread(original_image_path)

        # Check if either image is None
        if processed_image is None or original_img is None:
            return False

        # Check if dimensions match
        if processed_image.shape != original_img.shape:
            return False

        # Check image dimensions (height and width should be greater than 0)
        if processed_image.shape[0] <= 0 or processed_image.shape[1] <= 0:
            return False

        # Check the number of channels (expecting 3 for BGR/RGB images)
        if len(processed_image.shape) < 3 or processed_image.shape[2] != 3:
            return False

        # Check for significant changes in overall image content # TODO: check
        diff = cv2.absdiff(processed_image, original_img)
        non_zero_count = np.count_nonzero(diff)
        if non_zero_count / processed_image.size > 0.5:  # If more than 50% of pixels changed
            return False

        # If all checks pass, the image is considered valid
        return True

    except Exception as e:
        #logging.error(f"Error in check_image_integrity for {original_image_path}: {str(e)}")
        print(f"Error in check_image_integrity for {original_image_path}: {str(e)}")
        return False

