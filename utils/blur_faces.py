"""
TODO: add descriptions
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
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

    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.show()


def blur_face(image): # TODO: image_path or image?
    """
    Function to blur faces in an image using OpenCV.

    WARNING: This function attempts to blur faces in the given image automatically.
    However, due to the variability in image conditions and face orientations,
    successful face blurring cannot be guaranteed for every image.
    Please review the processed images carefully. Manual intervention may be required
    for images where faces are not adequately detected or blurred.

    :param image: image to process.
    :return: image.
    """

    # FIXME: update script

    '''
    try:
        # OpenCV reads images by default in BGR format
        image = cv2.imread(image_path)

        # Check if image was loaded successfully
        if image is None:
            print(f"Error loading image at {image_path}")
            return

        # Converting BGR image into a RGB image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Plotting the original image
        #plot_images(image)

        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt.xml') # TODO: check
        face_detect = cv2.CascadeClassifier(cascade_path)
        face_data = face_detect.detectMultiScale(image, 1.3, 5) # TODO: check

        # Draw rectangle around the faces which is our region of interest (ROI)
        for (x, y, w, h) in face_data:
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = image[y:y + h, x:x + w]
            # applying a gaussian blur over this new rectangle area
            roi = cv2.GaussianBlur(roi, (23, 23), 30)
            # impose this blurred image on original image to get final image
            image[y:y + roi.shape[0], x:x + roi.shape[1]] = roi

        # Display the output
        #plot_images(image)

    except cv2.error as e:
        print(f"OpenCV error processing image at {image_path}: {e}")
        return None

    return image
    '''

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use both frontal and profile face cascades for better detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Combine detected faces and profiles
        all_faces = np.vstack((faces, profiles)) if len(profiles) > 0 else faces

        for (x, y, w, h) in all_faces:
            # Increase the detected area slightly
            x, y = max(0, x - 10), max(0, y - 10)
            w, h = min(w + 20, image.shape[1] - x), min(h + 20, image.shape[0] - y)

            face_roi = image[y:y+h, x:x+w]

            # Apply a stronger blur
            blurred_face = cv2.GaussianBlur(face_roi, (33, 33), 30)

            # Blend the blurred face with the original to create a more natural look
            alpha = 0.8
            blended_face = cv2.addWeighted(blurred_face, alpha, face_roi, 1-alpha, 0)

            image[y:y+h, x:x+w] = blended_face

        return image

    except Exception as e:
        #logging.error(f"Error in blur_face: {str(e)}")
        print(f"Error in blur_face: {str(e)}")
        return image  # Return original image if blurring fails
