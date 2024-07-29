"""
TODO: add descriptions
"""

import os
import json


def create_json(categories, images, annotations, output_folder, name="labels"):
    """
    Creates a JSON file for the annotations.

    :param categories: List of categories for the annotations.
    :param images: List of images with their IDs and filenames.
    :param annotations: List of annotations with their bounding boxes.
    :param output_folder: Folder to save the JSON file.
    :param name: Name of the JSON file (default is "labels").
    :return: None
    """
    data = {
        "categories": categories,
        "images": images,
        "annotations": annotations
    }

    with open(os.path.join(output_folder, f"{name}.json"), "w") as f:
        json.dump(data, f, indent=4)


def format_json_for_mediapipe(json_data): # TODO: update name (?)
    """
    Formats the JSON data for use with MediaPipe.

    :param json_data: JSON data to format.
    :return: Formatted JSON data.
    """
    categories = json_data["categories"]
    images = json_data["images"]
    annotations = json_data["annotations"]

    # Add background category
    background_exists = False
    background_key_correct = False
    key_0_exists = False

    background = {"id": 0, "name": "background"}

    for category in categories:
        if category["name"] == "background":
            background_exists = True
            if category["id"] == 0:
                background_key_correct = True
            else:
                return "Error: 'background' category exists but does not have key 0." # TODO: check error type
        if category["id"] == 0 and category["name"] != "background":
            key_0_exists = True

    if key_0_exists and not background_key_correct:
        return "Error: Another category with key 0 exists that is not 'background'." # TODO: check error type

    if not background_exists:
        categories.insert(0, background)

    # Now prune some of the data # TODO: do I really need that?
    for category in categories:
        category.pop("supercategory", None)

    for image in images:
        image.pop("license", None)
        image.pop("flickr_url", None)
        image.pop("coco_url", None)
        image.pop("date_captured", None)

    for annotation in annotations:
        annotation.pop("segmentation", None)
        annotation.pop("area", None)
        annotation.pop("iscrowd", None)
        annotation.pop("attributes", None)

    # TODO: which format is used in json_data? x_min, y_min, x_max, y_max or x-top left, y-top left, width, height?

    return categories, images, annotations


def process_json_annotations(annotation_path): # TODO: update name (?)
    """
    Process JSON annotations for use with MediaPipe.

    :param annotation_path: Path to the JSON annotations.
    :return: Formatted JSON data.
    """
    with open(annotation_path, "r") as f:
        json_data = json.load(f)

    return format_json_for_mediapipe(json_data)


def update_annotations(categories, images, annotations, n_images, n_annotations): # TODO: update name (?)
    """
    Update the IDs for the images and annotations.

    :param categories:
    :param images:
    :param annotations:
    :param n_images:
    :param n_annotations:
    :return:
    """
    updated_categories = categories
    updated_images = []
    updated_annotations = []

    # TODO: check
    for image in images:
        image["id"] += n_images
        updated_images.append(image)

    # TODO: check
    for annotation in annotations:
        annotation["id"] += n_annotations
        annotation["image_id"] += n_images
        updated_annotations.append(annotation)

    return updated_categories, updated_images, updated_annotations
