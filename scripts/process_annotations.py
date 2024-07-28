"""
TODO: add descriptions
"""

import os
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
        json_path = os.path.join(output_folder, f"{num}_{monument}")
        monument_categories, monument_images, monument_annotations = process_json_annotations(json_path)

        n_images = len(images)
        n_annotations = len(annotations)

        monument_categories, monument_images, monument_annotations = update_annotations(
            monument_categories, monument_images, monument_annotations, n_images, n_annotations)

        # Append the categories, images, and annotations
        categories.extend(monument_categories)
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
            json_path = os.path.join(output_folder, f"{num}_{monument}")
            os.remove(json_path) # TODO: check


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
    process_all_json_annotations(annotations_path, output_folder, monuments)

    # Merge all JSON annotations
    merge_all_json_annotations(annotations_path, output_folder, monuments)


if __name__ == '__main__':
    main()
