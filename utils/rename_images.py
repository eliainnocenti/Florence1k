"""
This script renames all images in a specified directory to a standard format.

Functions:
----------
1. rename_images(monument_path, monument)
    Rename images in the specified monument directory.

Dependencies:
-------------
- os
- logging
- PIL.Image
- utils.check

Usage:
------
To run this script, ensure that the required libraries are installed and the data directory is correctly set.

Author:
-------
Elia Innocenti
"""

import os
import logging
from PIL import Image

from utils.check import check_directory

# Handling DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = 200000000 # TODO: check


def rename_images(monument_path, monument):
    """
    Function to rename images in the specified monument directory.

    :param monument_path: path to the directory containing images.
    :param monument: name of the monument.
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
