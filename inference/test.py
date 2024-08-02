"""
This script tests object detection on images using a TensorFlow Lite model trained on the Florence1K dataset.

Functions:
----------
1. prepare_image(image_path, input_shape)
    Prepare an image for object detection.

2. detect_objects(image_path, interpreter, input_details, output_details)
    Detect objects in an image.

3. test_images(images_path, monuments, confidence_thresholds, interpreter, input_details, output_details,
    n_images_per_monument=10, max_boxes=10, type='test_set')
    Test images for object detection.

Dependencies:
-------------
- numpy
- tensorflow
- cv2
- os
- random

Usage:
------
To run this script, ensure that the required libraries are installed and the data directory is correctly set.

Author:
-------
Elia Innocenti
"""

import numpy as np
import tensorflow as tf
import cv2
import os
import random

from utils.plot_images import plot_bounding_boxes

# FIXME: fix bounding boxes visualization
# TODO: study parameters and results

base_path = '../../../Data/'
florence1k_path = os.path.join(base_path, 'datasets', 'florence1k')

download_path = "/Users/eliainnocenti/Downloads"


def prepare_image(image_path, input_shape):
    """

    :param image_path:
    :param input_shape:
    :return:
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[1], input_shape[2]))
    img = img / 255.0  # Normalize # TODO: do I have to normalize?
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)


def detect_objects(image_path, interpreter, input_details, output_details):
    """
    Detect objects in an image.

    :param image_path: Path to the image to process.
    :param interpreter: TensorFlow Lite interpreter.
    :param input_details: Input details.
    :param output_details: Output details.
    :return:
    """
    input_shape = input_details[0]['shape']
    img = prepare_image(image_path, input_shape)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Get outputs
    boxes = interpreter.get_tensor(output_details[0]['index'])
    class_scores = interpreter.get_tensor(output_details[1]['index'])

    return boxes, class_scores


def test_images(images_path, monuments, confidence_thresholds, interpreter, input_details, output_details,
                n_images_per_monument=10, max_boxes=10, type='test_set'):
    """
    Test images for object detection.

    :param images_path: Path to the images directory
    :param monuments: Dictionary of monument names
    :param confidence_thresholds: Confidence thresholds for object detection
    :param n_images_per_monument: Number of images per monument to process
    :param max_boxes: Maximum number of boxes to display
    :param type: Type of images to test ('test_set' or 'personal_images')
    :return: None
    """

    # Personal images
    # image_path = os.path.join(download_path, 'IMG_4418.jpg')
    # image_path = os.path.join(download_path, 'eiffel.jpg')
    # image_path = os.path.join(download_path, 'IMG_5091.jpg')

    test_images = []

    if type == 'test_set':
        test_file_path = os.path.join("../annotations/object_detection/COCO/sets/test/", "test.txt")
        with open(test_file_path, 'r') as file:
            test_set = [x.strip() for x in file.readlines()]

        for num, monument in monuments.items():
            monument_path = os.path.join(images_path, f"{num}. {monument}")
            monument_images = [img for img in test_set if img.startswith(f"florence_{monument}")]

            if n_images_per_monument > 0:
                monument_images = random.sample(monument_images, min(n_images_per_monument, len(monument_images)))

            for image_name in monument_images:
                image_path = os.path.join(monument_path, image_name)
                test_images.append(image_path)

    elif type == 'personal_images':
        print("Not implemented yet.")
        return

    else:
        print("Invalid input. Please try again.")
        return

    for image_path in test_images:
        boxes, class_scores = detect_objects(image_path, interpreter, input_details, output_details)

        # Post-processing
        class_ids = np.argmax(class_scores[0], axis=1)
        confidences = np.max(class_scores[0], axis=1)

        if confidence_thresholds is None:
            print("Confidence thresholds not provided.")
            return

        if type == 'test_set':
            monument_num = image_path.split('/')[-2].split(' ')[0]
            confidence_threshold = confidence_thresholds[monument_num]
        elif type == 'personal_images':
            confidence_threshold = confidence_thresholds["0"]

        # Filter based on confidence threshold
        mask = confidences > confidence_threshold
        filtered_boxes = boxes[0][mask]
        filtered_class_ids = class_ids[mask]
        filtered_confidences = confidences[mask]

        # Apply non-max suppression (you might need to implement this)
        # For simplicity, let's just take the top max_boxes
        top_indices = np.argsort(filtered_confidences)[-max_boxes:]

        # Load the original image
        original_image = cv2.imread(image_path)
        height, width = original_image.shape[:2]

        bboxes = []

        print(f"Processing image: {image_path}")

        # Print results and draw boxes
        for i in top_indices:
            # Convert normalized coordinates to pixel coordinates
            xmin, ymin, b_width, b_height = filtered_boxes[i]       # xmin, ymin, xmax, ymax = filtered_boxes[i]
            xmin = int((xmin + 1) * width / 2)
            ymin = int((ymin + 1) * width / 2)
            b_width = int((b_width + 1) * height / 2)               # ymin = int((ymin + 1) * height / 2)
            b_height = int((b_height + 1) * height / 2)             # ymax = int((ymax + 1) * height / 2)

            print(f"Detection {i + 1}:")
            print(f"  Class: {filtered_class_ids[i]}")
            print(f"  Confidence: {filtered_confidences[i]:.2f}")
            # print(f"  Bounding box (pixel coordinates): [{xmin}, {ymin}, {xmax}, {ymax}]")
            print(f"  Bounding box (pixel coordinates): [{xmin}, {ymin}, {b_width}, {b_height}]")

            bboxes.append([xmin, ymin, b_width, b_height])

        #plot_bounding_boxes(original_image, bboxes, filtered_confidences, filtered_class_ids)

        # Save the image with bounding boxes
        # cv2.imwrite('output_image.jpg', original_image)
        # print("Output image saved as 'output_image.jpg'")


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

    confidence_thresholds = { # TODO: update with the correct values
        "1.": 0.5, "2.": 0.5, "3.": 0.0,  "4.": 0.0,  "5.": 0.0,  "6.": 0.0,
        "7.": 0.0, "8.": 0.0, "9.": 0.0, "10.": 0.0, "11.": 0.0, "12.": 0.0
    }

    images_path = os.path.join(florence1k_path, 'images')

    # Load the TFLite model
    model = input("Which model do you want to use? (model.tflite/model_fp16.tflite): ")
    if model == 'model.tflite':
        interpreter = tf.lite.Interpreter(model_path="../models/1/model.tflite")
    elif model == 'model_fp16.tflite':
        interpreter = tf.lite.Interpreter(model_path="../models/1/model_fp16.tflite")
    else:
        print("Invalid input. Please try again.")
        return

    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    type = input("Test set or personal images? (test_set/personal_images): ")
    if type == 'test_set':
        number = input("Number of images per monument: ")
        test_images(images_path, monuments, confidence_thresholds=confidence_thresholds,
                    interpreter=interpreter, input_details=input_details, output_details=output_details,
                    n_images_per_monument=int(number), max_boxes=10, type='test_set')
    elif type == 'personal_images':
        confidence_thresholds = {"0": 0.5} # TODO: update
        test_images(download_path, monuments, confidence_thresholds=confidence_thresholds,
                    interpreter=interpreter, input_details=input_details, output_details=output_details,
                    max_boxes=10, type='personal_images')
    else:
        print("Invalid input. Please try again.")
        return


if __name__ == '__main__':
    main()
