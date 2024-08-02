"""
This script contains functions to process the annotations for the Florence1K dataset.

Functions:
----------
1. process_all_json_annotations(annotations_path, output_folder, monuments)
    Process all JSON annotations for the specified monuments.

2. merge_all_json_annotations(annotations_path, output_folder, monuments)
    Merge all JSON annotations into a single JSON file.

3. split_train_val_test(monuments, train_percent=0.6, val_percent=0.3, test_percent=0.1, type='json')
    Split the annotations into training, validation, and test sets.

4. split_annotations(type='json')
    Split the annotations into training, validation, and test sets.

Dependencies:
-------------
- os
- json
- utils.annotations

Usage:
------
To run this script, ensure that the required libraries are installed and the data directory is correctly set.

Author:
-------
Elia Innocenti
"""

import os
import json

from utils.annotations import create_json
from utils.annotations import process_json_annotations
from utils.annotations import update_annotations

base_path = '../../../Data/'
florence1k_path = os.path.join(base_path, 'datasets', 'florence1k')


def process_all_json_annotations(annotations_path, output_folder, monuments):
    """


    :param annotations_path:
    :param output_folder:
    :param monuments:
    :return:
    """
    for num, monument in monuments.items():
        # Process the JSON annotations for each monument
        annotation_path = os.path.join(annotations_path, f"{num}. {monument}", "COCO")
        json_path = os.path.join(annotation_path, "annotations", "instances_default.json") # TODO: check path
        categories, images, annotations = process_json_annotations(json_path)

        # Create a new JSON file for each monument
        if not os.path.exists(os.path.join(output_folder, f"{num}_{monument}.json")):
            create_json(categories, images, annotations, output_folder, f"{num}_{monument}")
        else:
            answer = input(f"File {num}_{monument}.json already exists. Do you want to overwrite it? (y/n): ")
            if answer.lower() == "y":
                create_json(categories, images, annotations, output_folder, f"{num}_{monument}")


def merge_all_json_annotations(annotations_path, output_folder, monuments):
    """
    Merge all JSON annotations into a single JSON file.

    :param annotations_path: Path to the directory containing the JSON annotations.
    :param output_folder: Folder to save the merged JSON file.
    :param monuments: Dictionary of monument names with their corresponding numbers.
    :return: None.
    """
    categories = []
    images = []
    annotations = []

    for num, monument in monuments.items():
        # Process the JSON annotations for each monument
        json_path = os.path.join(output_folder, f"{num}_{monument}.json")
        monument_categories, monument_images, monument_annotations = process_json_annotations(json_path)

        n_images = len(images)
        n_annotations = len(annotations)

        monument_categories, monument_images, monument_annotations = update_annotations(
            monument_categories, monument_images, monument_annotations, n_images, n_annotations)

        # Append the categories, images, and annotations
        categories = monument_categories
        images.extend(monument_images)
        annotations.extend(monument_annotations)

    # Create a new JSON file for all the monuments
    if not os.path.exists(os.path.join(output_folder, "labels.json")):
        create_json(categories, images, annotations, output_folder, "labels")
    else:
        answer = input("File labels.json already exists. Do you want to overwrite it? (y/n): ")
        if answer.lower() == "y":
            create_json(categories, images, annotations, output_folder, "labels")

    # Ask to remove the individual JSON files
    remove_individual_files = input("Do you want to remove the individual JSON files? (y/n): ")
    if remove_individual_files.lower() == "y":
        for num, monument in monuments.items():
            json_path = os.path.join(output_folder, f"{num}_{monument}.json")
            os.remove(json_path) # TODO: check


def split_train_val_test(monuments, train_percent=0.6, val_percent=0.3, test_percent=0.1, type='json'): # TODO: update signature
    """
    TODO: update docstring
    """
    percent = train_percent + val_percent + test_percent
    if abs(1 - percent) > 0.000001:
        print("Error: Train, validation and test percentages must sum up to 1")
        return

    # Check if txt files are already created
    path = os.path.join("../annotations", "object_detection", "COCO")
    if os.path.exists(os.path.join(path, 'sets', 'train', 'train.txt')) or \
       os.path.exists(os.path.join(path, 'sets', 'validation', 'val.txt')) or \
       os.path.exists(os.path.join(path, 'sets', 'test', 'test.txt')):
        print(f"Train, validation and test sets already created.")
        return

    images = []

    if type == 'json':
        labels_file = os.path.join(path, "labels.json")
        if not os.path.exists(labels_file):
            print(f"Error: Labels file not found: {labels_file}")
            return
        with open(labels_file, 'r') as file:
            labels_json = json.load(file)

        images = labels_json['images']

    if not images:
        print("Error: No images found.")
        return

    train_size = int(len(images) * train_percent)
    val_size = int(len(images) * val_percent)
    test_size = len(images) - train_size - val_size

    n_images_per_monument = 100
    n_monuments = len(monuments)

    train_size_rel = int(train_size / n_monuments)
    val_size_rel = int(val_size / n_monuments)
    test_size_rel = n_images_per_monument - train_size_rel - val_size_rel

    train_set, val_set, test_set = [], [], []

    for num, monument in monuments.items():
        # Calculate the offset for the current monument
        offset = (int(num) - 1) * 100

        # Append the training images
        train_set += images[offset:offset + train_size_rel]

        # Append the validation images
        val_set += images[offset + train_size_rel:offset + train_size_rel + val_size_rel]

        # Append the test images
        test_set += images[
                    offset + train_size_rel + val_size_rel:offset + train_size_rel + val_size_rel + test_size_rel]

    print(f"Train set: {len(train_set)}, Validation set: {len(val_set)}, Test set: {len(test_set)}")

    with open(os.path.join(path, 'sets', 'train', 'train.txt'), 'w') as file:
        for image in train_set:
            file.write(f'{image['file_name']}\n')

    with open(os.path.join(path, 'sets', 'validation', 'val.txt'), 'w') as file:
        for image in val_set:
            file.write(f'{image['file_name']}\n')

    with open(os.path.join(path, 'sets', 'test', 'test.txt'), 'w') as file:
        for image in test_set:
            file.write(f'{image['file_name']}\n')


def split_annotations(type='json'): # TODO: update signature
    """
    # TODO: update docstring
    """
    train_set = []
    val_set = []
    test_set = []

    path = os.path.join("../annotations", "object_detection", "COCO")
    sets_dir = os.path.join(path, 'sets')

    with open(os.path.join(sets_dir, 'train', 'train.txt'), 'r') as file:
        for line in file:
            train_set.append(line.strip())

    with open(os.path.join(sets_dir, 'validation', 'val.txt'), 'r') as file:
        for line in file:
            val_set.append(line.strip())

    with open(os.path.join(sets_dir, 'test', 'test.txt'), 'r') as file:
        for line in file:
            test_set.append(line.strip())

    if type == 'json':
        annotations_dir = os.path.join('../annotations', 'object_detection', 'COCO')
        labels_file = os.path.join(annotations_dir, 'labels.json')
        if not os.path.exists(labels_file):
            print(f"Error: Labels file not found: {labels_file}")
            return
        with open(labels_file, 'r') as file:
            labels_json = json.load(file)
        categories = labels_json['categories']
        images = labels_json['images']
        annotations = labels_json['annotations']

        print(f"Number of images: {len(images)}")

        train_images = [image for image in images if image['file_name'] in train_set]
        val_images = [image for image in images if image['file_name'] in val_set]
        test_images = [image for image in images if image['file_name'] in test_set]

        print(f"Number of train images: {len(train_images)}")
        print(f"Number of val images: {len(val_images)}")
        print(f"Number of test images: {len(test_images)}")

        train_annotations = [annotation for annotation in annotations if
                             annotation['image_id'] in [image['id'] for image in train_images]]
        val_annotations = [annotation for annotation in annotations if
                           annotation['image_id'] in [image['id'] for image in val_images]]
        test_annotations = [annotation for annotation in annotations if
                            annotation['image_id'] in [image['id'] for image in test_images]]

        create_json(categories, train_images, train_annotations, os.path.join(sets_dir, 'train'), name='train')
        create_json(categories, val_images, val_annotations, os.path.join(sets_dir, 'validation'), name='val')
        create_json(categories, test_images, test_annotations, os.path.join(sets_dir, 'test'), name='test')

    else:
        print("Error: Invalid type of annotation")
        return


def main():
    """
    Main function to process the annotations.

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

    annotations_path = os.path.join(florence1k_path, 'annotations', 'object_detection')
    output_folder = os.path.join("../annotations", "object_detection", "COCO")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all JSON annotations
    #process_all_json_annotations(annotations_path, output_folder, monuments)

    # Merge all JSON annotations
    #merge_all_json_annotations(annotations_path, output_folder, monuments)

    # Split the annotations
    #split_train_val_test(monuments)
    #split_annotations()


if __name__ == '__main__':
    main()
