"""
This script contains utility functions to blur faces in an image using OpenCV's DNN face detector.

Functions:
----------
1. download_model_files(download_dir=".")
    Download the required model files for face detection.

2. plot_images(img)
    Plot images using matplotlib.

3. blur_face(image)
    Blur faces in an image using OpenCV's DNN face detector.

Dependencies:
-------------
- os
- cv2
- matplotlib.pyplot
- urllib.request

Usage:
------
To run this script, ensure that the required libraries are installed and the data directory is correctly set.

Author:
-------
Elia Innocenti
"""

import os
import cv2
import matplotlib.pyplot as plt
import urllib.request


def download_model_files(download_dir="."):
    """
    Function to download the required model files.

    :param download_dir: Directory to download the model files to.
    :return: None.
    """
    #base_url = "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/"
    base_url = "https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/models/"
    files = {
        "opencv_face_detector_uint8.pb": base_url + "opencv_face_detector_uint8.pb",
        "opencv_face_detector.pbtxt": base_url + "opencv_face_detector.pbtxt"
    }

    # Ensure the download directory exists
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    print("Downloading model files...")
    for file_name, url in files.items():
        try:
            print(f"Downloading {file_name} to {download_dir}...")
            file_path = os.path.join(download_dir, file_name)
            urllib.request.urlretrieve(url, file_path)
            print(f"{file_name} downloaded successfully to {file_path}.")

        except urllib.error.URLError as e:
            print(f"Error downloading {file_name}: {e}")
            print(f"Please download the file manually from: {url}")
            print(f"and place it in the current working directory: {os.getcwd()}")

    print("Download process completed.")


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


def blur_face(image):
    """
    Function to blur faces in an image using OpenCV's DNN face detector.

    WARNING: This function attempts to blur faces in the given image automatically.
    However, due to the variability in image conditions and face orientations,
    successful face blurring cannot be guaranteed for every image.
    Please review the processed images carefully. Manual intervention may be required
    for images where faces are not adequately detected or blurred.

    :param image: image to process (numpy array).
    :return: image with blurred faces.
    """
    try:
        model_file = "../utils/opencv_face_detector_uint8.pb"
        config_file = "../utils/opencv_face_detector.pbtxt"

        # Check if model files exist, if not, download them
        if not (os.path.exists(model_file) and os.path.exists(config_file)):
            print("Model files not found. Downloading them now...")
            download_model_files("../utils")  # Download to the 'utils' directory

        # Check again if files exist after download attempt
        if not (os.path.exists(model_file) and os.path.exists(config_file)):
            print("Error: Model files are still missing. Unable to proceed with face blurring.")
            return image  # Return the original image if files are missing

        try:
            # Load the DNN model
            net = cv2.dnn.readNetFromTensorflow(model_file, config_file) # FIXME

        except cv2.error as e:
            print(f"Error loading the DNN model: {str(e)}")
            print("Please ensure you have the following files in your working directory:")
            print(f"1. {model_file}")
            print(f"2. {config_file}")
            print("You can download these files manually from:")
            print("https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/models/")
            print("https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/")
            return image  # Return the original image if model loading fails

        original_image = image.copy()  # Keep a copy of the original image
        height, width = image.shape[:2]

        '''debug'''
        plot_images(original_image)

        # Calculate the scale factor
        scale_factor = 1.0
        max_size = max(height, width)
        if max_size > 1000:
            scale_factor = 1000.0 / max_size

        # Resize image for face detection only if necessary
        if scale_factor < 1.0:
            resized_image = cv2.resize(original_image, (int(width * scale_factor), int(height * scale_factor)))
        else:
            resized_image = original_image

        '''debug'''
        plot_images(resized_image)

        # Create a blob from the resized image
        blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), [104, 117, 123], False, False)

        # Set the blob as input and perform forward pass
        net.setInput(blob)
        detections = net.forward()

        # Confidence threshold for detections
        conf_threshold = 0.7 # TODO: check

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                # Scale the bounding box coordinates back to the original image size
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)

                # Ensure coordinates are within image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                # Extract the region of interest (face) from the original image
                face_roi = original_image[y1:y2, x1:x2]

                # Calculate the size of blur kernel based on face size
                kernel_size = min(face_roi.shape[:2])
                kernel_size = (kernel_size // 7) | 1  # Ensure kernel size is odd

                # Apply a stronger blur
                blurred_face = cv2.GaussianBlur(face_roi, (kernel_size, kernel_size), 30)

                # Blend the blurred face with the original for a more natural look
                alpha = 0.8
                blended_face = cv2.addWeighted(blurred_face, alpha, face_roi, 1 - alpha, 0)

                # Replace the face region with the blended face in the original image
                original_image[y1:y2, x1:x2] = blended_face

        '''debug'''
        plot_images(original_image)

        return original_image

    except Exception as e:
        print(f"Error in blur_face: {str(e)}")
        return image  # Return original image if blurring fails
