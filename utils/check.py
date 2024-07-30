"""
This script contains utility functions for checking images integrity and properties.

Functions:
----------
1. get_image_size(image_path)
    Get the size of an image.

2. find_large_images(directory, max_pixels)
    Find images in a directory that exceed a certain number of pixels.

3. check_image_integrity(processed_image, original_image_path)
    Check the integrity of a processed image against the original image.

4. is_jpg(filename)
    Check if a file is a JPEG image.

5. check_directory(dir_path)
    Check if all files in a directory are JPEG images.

6. count_images(images_path, monuments)
    Count the number of images in each directory.

7. check_all_directories_jpeg(images_path, monuments)
    Check if all directories contain only JPEG images.

8. check_all_directories_size(images_path, monuments)
    Check if all images in the directories are below a certain size.

Dependencies:
-------------
- os
- cv2
- numpy
- PIL.Image

Usage:
------
To run this script, ensure that the required libraries are installed and the data directory is correctly set.

Author:
-------
Elia Innocenti
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


def is_jpg(filename):
    """
    Function to check if a file is a JPEG image.

    :param filename: path to the file.
    :return: True if the file is a JPEG image, False otherwise.
    """
    try:
        with Image.open(filename) as img:
            return img.format == 'JPEG'
    except IOError:
        return False


def check_directory(dir_path):
    """
    Function to check if all files in a directory are JPEG images.

    :param dir_path: path to the directory.
    :return: True if all files are JPEG images, False otherwise.
    """
    valid = True
    for file in os.listdir(dir_path):
        if not is_jpg(os.path.join(dir_path, file)):
            if file.endswith('.DS_Store'): # TODO: check
                continue
            else:
                print(f"File {os.path.join(dir_path, file)} is not a JPEG image.")
                valid = False

    return valid


def count_images(images_path, monuments):
    """
    Count the number of images in each directory.

    :param images_path: Path to the directory containing monument subdirectories.
    :param monuments: Dictionary of monument names with their corresponding numbers.
    :return: None.
    """
    for num, monument in monuments.items():
        monument_path = os.path.join(images_path, f"{num}. {monument}")
        images = [img for img in os.listdir(monument_path) if is_jpg(os.path.join(monument_path, img))]
        print(f"Monument {monument} has {len(images)} images.")


def check_all_directories_jpeg(images_path, monuments): # TODO: update name (?)
    """
    Check if all directories contain only JPEG images.

    :param images_path: Path to the directory containing monument subdirectories.
    :param monuments: Dictionary of monument names with their corresponding numbers.
    :return: None.
    """
    for num, monument in monuments.items():
        monument_path = os.path.join(images_path, f"{num}. {monument}")
        if check_directory(monument_path):
            print(f'All images in {monument} are JPEG images.')
        else:
            print(f'Not all images in {monument} are JPEG images.')


def check_all_directories_size(images_path, monuments): # TODO: update name (?)
    """
    Check if all images in the directories are below a certain size.

    :param images_path:
    :param monuments:
    :return:
    """
    max_pixels = 178956970  # Example pixel limit

    for num, monument in monuments.items():
        monument_path = os.path.join(images_path, f"{num}. {monument}")
        large_images = find_large_images(monument_path, max_pixels)
        if len(large_images) > 0:
            print(f"Monument {monument} has {len(large_images)} images exceeding {max_pixels} pixels.")
            for image in large_images:
                print(f"Image {image} exceeds {max_pixels} pixels.")
        else:
            print(f"All images in {monument} are below {max_pixels} pixels.")
