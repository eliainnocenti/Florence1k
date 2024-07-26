"""
TODO: add descriptions
This script processes images for the Florence1k dataset, including face blurring,
image renaming, and metadata removal.
"""

import os
import cv2
from tqdm import tqdm
import numpy as np
from utils.blur_faces import blur_face
from utils.rename_images import rename_images
from utils.rename_images import is_jpg
from utils.rename_images import check_directory
from utils.remove_metadata import remove_metadata

base_path = '../../../Data/'
florence1k_path = os.path.join(base_path, 'datasets', 'florence1k')


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
    #total_images = sum(len(os.listdir(os.path.join(images_path, f"{num}. {name}"))) for num, name in monuments.items())
    total_images = sum(len([f for f in os.listdir(os.path.join(images_path, f"{num}. {name}"))
                            if is_jpg(os.path.join(images_path, f"{num}. {name}", f))])
                       for num, name in monuments.items())

    with tqdm(total=total_images, desc="Blurring Faces") as progress_bar:
        for num, monument in monuments.items():
            monument_path = os.path.join(images_path, f"{num}. {monument}")

            for image in os.listdir(monument_path):
                image_path = os.path.join(monument_path, image)

                if not is_jpg(image_path):
                    #logging.warning(f"Skipping invalid image: {image_path}")
                    print(f"Skipping invalid image: {image_path}")
                    continue

                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        raise ValueError(f"Failed to load image: {image_path}")

                    blurred_img = blur_face(img)

                    if check_image_integrity(blurred_img, image_path):
                        cv2.imwrite(image_path, blurred_img) # If the image already exists, it will be overwritten
                        #cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    else:
                        print(f"Warning: Integrity check failed for {image_path}. Image not saved.")

                except cv2.error as e:
                    if "Invalid SOS parameters for sequential JPEG" in str(e):
                        #logging.warning(f"Invalid JPEG in {image_path}. Skipping.")
                        print(f"Invalid JPEG in {image_path}. Skipping.")
                    else:
                        #logging.error(f"OpenCV error processing image at {image_path}: {str(e)}")
                        print(f"OpenCV error processing image at {image_path}: {str(e)}")
                except Exception as e:
                    #logging.error(f"Error processing image at {image_path}: {str(e)}")
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
        rename_images(monument_path, monument, num)
        progress_bar.update(len(os.listdir(monument_path)))

    progress_bar.close()


def remove_all_metadata():
    # Implement metadata removal logic here
    pass


def check_all_directories(images_path, monuments):
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


def main():
    """
    Main function to process the data.

    :return: None.
    """
    monuments = {
        "1": "santamariadelfiore",
        #"2": "battisterosangiovanni",
        #"3": "campanilegiotto",
        #"4": "galleriauffizi",
        #"5": "loggialanzi",
        #"6": "palazzovecchio",
        #"7": "pontevecchio",
        #"8": "basilicasantacroce",
        #"9": "palazzopitti",
        #"10": "piazzalemichelangelo",
        #"11": "basilicasantamarianovella",
        #"12": "basilicasanminiato"
    }

    images_path = os.path.join(florence1k_path, 'images')

    # Blur faces in all images
    #blur_all_faces(images_path, monuments) # FIXME

    # Rename all images
    #rename_all_images(images_path, monuments)

    # debug # TODO: remove
    #count_images(images_path, monuments)
    #check_all_directories(images_path, monuments)

    # Remove metadata from all images
    #remove_all_metadata()


if __name__ == '__main__':
    main()
