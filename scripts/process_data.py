"""
This script processes images for the Florence1k dataset, including face blurring, image renaming, and extra metadata removal.

Functions:
----------
1. blur_all_faces(images_path, monuments)
    Blur faces in all images within the specified monuments.

2. rename_all_images(images_path, monuments)
    Rename all images within the specified monuments.

3. remove_all_extra_metadata(images_path, monuments)
    Remove extra metadata from all images.

4. list_all_images(images_path, monuments, output_file)
    List all images and save the data to a CSV file.

Dependencies:
-------------
- os
- cv2
- tqdm
- utils.blur_faces
- utils.rename_images
- utils.remove_metadata
- utils.check

Usage:
------
To run this script, ensure that the required libraries are installed and the data directory is correctly set.

Author:
-------
Elia Innocenti
"""

import os
import cv2
from tqdm import tqdm

from utils.blur_faces import blur_faces
from utils.rename_images import rename_images
from utils.rename_images import check_directory
from utils.remove_extra_metadata import remove_extra_metadata
from utils.list_images import process_directory
from utils.list_images import create_csv
from utils.list_images import custom_sort_key
from utils.check import find_large_images
from utils.check import check_image_integrity
from utils.check import is_jpg
from utils.check import count_images
from utils.check import check_all_directories_jpeg
from utils.check import check_all_directories_size

base_path = '../../../Data/'
florence1k_path = os.path.join(base_path, 'datasets', 'florence1k')


def blur_all_faces(images_path, monuments):
    """
    Blur faces in all images within the specified monuments.

    WARNING: This function attempts to blur faces in the given image automatically.
    However, due to the variability in image conditions and face orientations,
    successful face blurring cannot be guaranteed for every image.
    Please review the processed images carefully. Manual intervention may be required
    for images where faces are not adequately detected or blurred.

    :param images_path: Path to the directory containing monument subdirectories.
    :param monuments: Dictionary of monument names with their corresponding numbers.
    :return: None.
    """
    total_images = sum(len([f for f in os.listdir(os.path.join(images_path, f"{num}. {name}"))
                            if is_jpg(os.path.join(images_path, f"{num}. {name}", f))])
                       for num, name in monuments.items())

    with tqdm(total=total_images, desc="Blurring Faces") as progress_bar:
        for num, monument in monuments.items():
            monument_path = os.path.join(images_path, f"{num}. {monument}")

            for image in os.listdir(monument_path):
                image_path = os.path.join(monument_path, image)

                if not is_jpg(image_path):
                    print(f"Skipping invalid image: {image_path}")
                    continue

                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        raise ValueError(f"Failed to load image: {image_path}")

                    blurred_img = blur_faces(img)

                    if check_image_integrity(blurred_img, image_path):
                        cv2.imwrite(image_path, blurred_img) # If the image already exists, it will be overwritten
                    else:
                        print(f"Warning: Integrity check failed for {image_path}. Image not saved.")

                except cv2.error as e:
                    if "Invalid SOS parameters for sequential JPEG" in str(e):
                        print(f"Invalid JPEG in {image_path}. Skipping.")
                    else:
                        print(f"OpenCV error processing image at {image_path}: {str(e)}")
                except Exception as e:
                    print(f"Error processing image at {image_path}: {str(e)}")

                progress_bar.update(1)


def rename_all_images(images_path, monuments):
    """
    Rename all images within the specified monuments.

    :param images_path: Path to the directory containing monument subdirectories.
    :param monuments: Dictionary of monument names with their corresponding numbers.
    :return: None.
    """
    total_images = 0

    # TODO: check progress bar
    # FIXME: implement Error handling

    for num, monument in monuments.items():
        monument_path = os.path.join(images_path, f"{num}. {monument}")
        images = [img for img in os.listdir(monument_path) if is_jpg(os.path.join(monument_path, img))]
        total_images += len(images)

    progress_bar = tqdm(total=total_images, desc="Renaming Images")

    for num, monument in monuments.items():
        monument_path = os.path.join(images_path, f"{num}. {monument}")
        rename_images(monument_path, monument)
        progress_bar.update(len(os.listdir(monument_path)))

    progress_bar.close()


def remove_all_extra_metadata(images_path, monuments): # TODO: check
    """
    Remove extra metadata from all images.

    :param images_path: Path to the directory containing monument folders.
    :param monuments: Dictionary of monument numbers and names.
    :return: None.
    """
    total_images = sum(len([f for f in os.listdir(os.path.join(images_path, f"{num}. {name}"))
                            if is_jpg(os.path.join(images_path, f"{num}. {name}", f))])
                       for num, name in monuments.items())

    with tqdm(total=total_images, desc="Removing Extra Metadata") as progress_bar:
        for num, monument in monuments.items():
            monument_path = os.path.join(images_path, f"{num}. {monument}")

            for image in os.listdir(monument_path):
                image_path = os.path.join(monument_path, image)

                if not is_jpg(image_path):
                    print(f"Skipping invalid image: {image_path}")
                    continue

                try:
                    remove_extra_metadata(image_path)

                except Exception as e:
                    print(f"Error processing image at {image_path}: {str(e)}")

                progress_bar.update(1)


def list_all_images(images_path, monuments, output_file):
    """
    List all images and save the data to a CSV file.

    :param images_path: Path to the directory containing monument folders.
    :param monuments: Dictionary of monument numbers and names.
    :param output_file: Path to the output CSV file.
    :return: None.
    """
    all_image_data = []
    total_monuments = len(monuments)

    # Create a progress bar for processing monuments
    with tqdm(total=total_monuments, desc="Processing monuments", unit="monument") as pbar:
        for num, monument in monuments.items():
            monument_path = os.path.join(images_path, f"{num}. {monument}")

            # Process the directory and update the progress bar
            monument_data = process_directory(monument_path)
            all_image_data.extend(monument_data)

            pbar.update(1)
            pbar.set_postfix({"Current": monument[:20]}, refresh=True)

    # Sort by folder name and then by file name
    #all_image_data = sorted(all_image_data, key=lambda x: (x['folder_name'], x['file_name']))
    all_image_data = sorted(all_image_data, key=custom_sort_key)

    create_csv(all_image_data, output_file)

    print(f"\nProcessed {total_monuments} monuments. Data saved to {output_file}")



def main():
    """
    Main function to process the data.

    :return: None.
    """
    monuments = {
        "1": "santamariadelfiore",
        "2": "battisterosangiovanni",
        "3": "campanilegiotto",
        "4": "galleriauffizi",
        "5": "loggialanzi",
        "6": "palazzovecchio",
        "7": "pontevecchio",
        "8": "basilicasantacroce",
        "9": "palazzopitti",
        "10": "piazzalemichelangelo",
        "11": "basilicasantamarianovella",
        "12": "basilicasanminiato"
    }

    images_path = os.path.join(florence1k_path, 'images')

    # Blur faces in all images
    #blur_all_faces(images_path, monuments) # FIXME

    # debug # TODO: remove
    '''
    test_image_path = os.path.join(images_path, '1. santamariadelfiore', 'florence_santamariadelfiore_0001.jpg')
    test_image = cv2.imread(test_image_path)
    blurred_test_image = blur_faces(test_image)
    '''

    # Rename all images
    #rename_all_images(images_path, monuments)

    # debug # TODO: remove
    '''
    print("\nNumber of images:")
    count_images(images_path, monuments)
    print("\nJPEG check:")
    check_all_directories_jpeg(images_path, monuments)
    print("\nSize check:")
    check_all_directories_size(images_path, monuments)
    '''

    # Remove extra metadata from all images
    #remove_all_extra_metadata(images_path, monuments)

    # List all images and save the data to a CSV file
    output_file = os.path.join('../images', 'images.csv')
    #list_all_images(images_path, monuments, output_file)


if __name__ == '__main__':
    main()
