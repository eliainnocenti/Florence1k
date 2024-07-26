"""
TODO: add descriptions
"""

import numpy as np
import cv2
import os
import logging
from PIL import Image

# Handling DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = 200000000 # TODO: check


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


def rename_images(monument_path, monument, monument_number):
    """
    Function to rename images in the specified monument directory.

    :param monument_path: path to the directory containing images.
    :param monument: name of the monument.
    :param monument_number: number of the monument.
    :return: None.
    """
    # Setup logging
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    log_file_path = os.path.join(logs_dir, 'rename_images.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Check if the directory exists
    if not os.path.exists(monument_path):
        print(f"Directory '{monument_path}' does not exist.")
        logging.error(f"Directory '{monument_path}' does not exist.")
        return

    # Check if the directory contains only JPEG images
    if not check_directory(monument_path):
        print(f"Directory '{monument_path}' contains non-JPEG images. Skipping rename.")
        logging.error(f"Directory '{monument_path}' contains non-JPEG images. Skipping rename.")
        return

    # Get the list of files in the directory
    images = os.listdir(monument_path)

    # Remove hidden files
    images = [img for img in images if not img.startswith('.')]

    # Sort the files to ensure consistent naming
    images.sort()

    # FIXME: implement Error handling and Name conflict resolution

    # TODO: check if the image had already been renamed
    # the name must be in the format florence_{monument}_{count:04d}.jpg
    # if the image has already been renamed, skip the renaming process
    # if the image has not been renamed, rename it

    # Initialize the counter
    count = 1

    '''
    # Rename all images to a standard format
    for i, image in enumerate(images):
        # Ensure that every image is a jpg file
        if not is_jpg(os.path.join(monument_path, image)):
            logging.warning(f"File {os.path.join(monument_path, image)} is not a JPEG image. Skipping rename.")
            continue

        new_name = f"{str(i + 1)}.jpg"

        old_path = os.path.join(monument_path, image)
        new_path = os.path.join(monument_path, new_name)

        if os.path.exists(new_path):
            logging.warning(f"File {new_path} already exists. Skipping rename to avoid overwrite.")
            continue

        os.rename(old_path, new_path)
    '''

    # Iterate over all files in the directory
    for image in images:
        # Check if the file is an image
        if image.lower().endswith('.jpg'):
            # Construct the new file name
            new_name = f"florence_{monument}_{count:04d}.jpg"

            # Construct the full path for the image
            old_path = os.path.join(monument_path, image)
            new_path = os.path.join(monument_path, new_name)

            if os.path.exists(new_path):
                logging.warning(f"File {new_path} already exists. Skipping rename to avoid overwrite.")
                continue

            # Rename the image file
            os.rename(old_path, new_path)

            # Increment the counter
            count += 1


    print(f"Completed renaming. Processed {count - 1} images in '{monument_path}'.")
    logging.info(f"Completed renaming. Processed {count - 1} images in '{monument_path}'.")
