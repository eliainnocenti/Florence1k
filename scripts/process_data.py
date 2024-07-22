"""
TODO: add descriptions
"""

import os
import cv2
import tqdm
from utils.blur_faces import blur_face
from utils.rename_images import rename_images
from utils.remove_metadata import remove_metadata

base_path = '../../../Data/'
florence1k_path = os.path.join(base_path, 'datasets', 'florence1k')


def blur_all_faces(images_path, monuments):
    """

    :param images_path:
    :param monuments:
    :return:
    """
    total_images = sum(
        [len(os.listdir(os.path.join(images_path, str(str(i) + '. ' + monument)))) for i, monument in enumerate(monuments)])
    progress_bar = tqdm(total=total_images, desc="Processing Images") # TODO: update

    for i, monument in enumerate(monuments):
        monument_path = os.path.join(images_path, f"{i}. {monument}")

        for image in os.listdir(monument_path):
            image_path = os.path.join(monument_path, image)
            img = blur_face(image_path)

            # TODO: do I have to remove the original photo first?

            # Save the image
            cv2.imwrite(image_path, img)
            # cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            progress_bar.update(1)

    progress_bar.close()


def rename_all_images():
    return None


def remove_all_metadata():
    return None


def main():

    monuments = ['santamariadelfiore', 'battisterosangiovanni', 'campanilegiotto', 'galleriauffizi', 'loggialanzi',
                 'palazzovecchio', 'pontevecchio', 'basilicasantacroce', 'palazzopitti', 'piazzalemichelangelo',
                 'basilicasantamarianovella', 'basilicasanminiato']

    images_path = os.path.join(florence1k_path, 'images')

    blur_all_faces(images_path, monuments)


if __name__ == '__main__':
    main()
