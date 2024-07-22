"""
TODO: add descriptions
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def plotImages(img):
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

    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.show()


def blur_face(image_path):
    """
    Function to blur faces in an image using OpenCV.
    :param image_path: path to the image.
    :return: image.
    """
    # OpenCV reads images by default in BGR format
    image = cv2.imread(image_path)

    # Check if image was loaded successfully
    if image is None:
        print(f"Error loading image at {image_path}")
        return

    # Converting BGR image into a RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # plotting the original image
    #plotImages(image)

    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    face_data = face_detect.detectMultiScale(image, 1.3, 5)

    # Draw rectangle around the faces which is our region of interest (ROI)
    for (x, y, w, h) in face_data:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image[y:y + h, x:x + w]
        # applying a gaussian blur over this new rectangle area
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        # impose this blurred image on original image to get final image
        image[y:y + roi.shape[0], x:x + roi.shape[1]] = roi

    # Display the output
    #plotImages(image)

    return image
