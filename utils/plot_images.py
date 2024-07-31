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


def plot_bounding_boxes(img, boxes, class_scores, class_names):
    """
    Function to plot bounding boxes on an image.

    :param img: image to plot.
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

    for i in range(len(boxes)):
        box = boxes[i]
        class_score = class_scores[i]
        class_name = class_names[i]

        x1, x2, w, h = box
        y1 = x1 + w
        y2 = x2 + h
        img = cv2.rectangle(img, (x1, x2), (y1, y2), (0, 255, 0), 2)
        img = cv2.putText(img, f"{class_name} {class_score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
