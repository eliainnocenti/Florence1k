"""
This file contains the function to plot images using matplotlib.

Functions:
----------
1. plot_images(img)
    Plot images using matplotlib.

2. plot_bounding_boxes(img, boxes, class_scores, class_names)
    Plot bounding boxes on an image using matplotlib.

Dependencies:
-------------
- cv2
- matplotlib.pyplot

Usage:
------
To use this functions, ensure that the required libraries are installed.

Author:
-------
Elia Innocenti
"""

import cv2
import matplotlib.pyplot as plt
import os


def plot_images(img):
    """
    Function to plot images.

    :param img: image to plot.
    :return: None.
    """
    # Check if 'seaborn' style is available, otherwise use a default style
    if 'seaborn' in plt.style.available:
        plt.style.use('seaborn')
    else:
        # Use a default style, e.g., 'classic'
        plt.style.use('classic')

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def plot_bounding_boxes(img_path, boxes, class_names, class_scores=None, output_path=None):
    """
    Function to plot bounding boxes on an image.

    :param img_path: path to the image to plot.
    :param boxes: bounding boxes.
    :param class_scores: class scores.
    :param class_names: class names.
    :return: None.
    """
    # Check if 'seaborn' style is available, otherwise use a default style
    if 'seaborn' in plt.style.available:
        plt.style.use('seaborn')
    else:
        # Use a default style, e.g., 'classic'
        plt.style.use('classic')

    # Read the image to get its dimensions
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # Calculate the thickness proportional to the image size
    # Assuming 10 is perfect for an image of size 3024x4032
    base_thickness = 10
    reference_size = 3024 * 4032
    current_size = width * height
    thickness = int(base_thickness * (current_size / reference_size))

    # Ensure the thickness is at least 1
    thickness = max(2, thickness)

    for i in range(len(boxes)):
        box = boxes[i]
        class_name = class_names[i]

        x1, x2, w, h = map(int, box)  # Ensure coordinates are integers
        y1 = x1 + w
        y2 = x2 + h
        img = cv2.rectangle(img, (x1, x2), (y1, y2), (0, 255, 0), thickness)

        if output_path is None:
            if class_scores is None:
                img = cv2.putText(img, f"{class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                class_score = class_scores[i]
                img = cv2.putText(img, f"{class_name} {class_score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    if output_path is not None:
        # Extract the original image name and extension
        original_name, ext = os.path.splitext(os.path.basename(img_path))

        # Append the suffix "_bboxes" to the original image name
        new_image_name = f"{original_name}_bboxes{ext}"

        # Ensure the output path includes the modified image name
        output_path = os.path.join(output_path, new_image_name)

        # Check if the output path already exists
        if os.path.exists(output_path):
            print(f"Error: The file {output_path} already exists.")
            return

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the image
        cv2.imwrite(output_path, img)
        print(f"Saved image with bounding boxes to: {output_path}")
