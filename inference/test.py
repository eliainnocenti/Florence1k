"""
TODO: add file and function descriptions
"""

import numpy as np
import tensorflow as tf
import cv2
import os

from utils.plot_images import plot_bounding_boxes

# FIXME: fix bounding boxes visualization
# TODO: study parameters and results

base_path = '../../../Data/'
florence1k_path = os.path.join(base_path, 'datasets', 'florence1k')

download_path = "/Users/eliainnocenti/Downloads"

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="../models/1/model.tflite")
#interpreter = tf.lite.Interpreter(model_path="../models/1/model_fp16.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


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


def detect_objects(image_path):
    """

    :param image_path:
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


# Example usage with test set images
#image_path = os.path.join(florence1k_path, 'images', '1. santamariadelfiore', 'florence_santamariadelfiore_0099.jpg')
#image_path = os.path.join(florence1k_path, 'images', '2. battisterosangiovanni', 'florence_battisterosangiovanni_0095.jpg')
#image_path = os.path.join(florence1k_path, 'images', '6. palazzovecchio', 'florence_palazzovecchio_0093.jpg')

# Examples usage with personal images
#image_path = os.path.join(download_path, 'IMG_4418.jpg')
#image_path = os.path.join(download_path, 'eiffel.jpg')
image_path = os.path.join(download_path, 'IMG_5091.jpg')


boxes, class_scores = detect_objects(image_path)

# Post-processing
confidence_threshold = 0.1  # 0.5
max_boxes = 10

# Get the class with highest score for each box
class_ids = np.argmax(class_scores[0], axis=1)
confidences = np.max(class_scores[0], axis=1)

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

# Print results and draw boxes
for i in top_indices:
    # Convert normalized coordinates to pixel coordinates
    # FIXME: which format do I have in output? [xmin, ymin, xmax, ymax] or [xmin, ymin, width, height]?

    '''
    xmin, ymin, xmax, ymax = filtered_boxes[i]
    xmin = int((xmin + 1) * width / 2)
    xmax = int((xmax + 1) * width / 2)
    ymin = int((ymin + 1) * height / 2)
    ymax = int((ymax + 1) * height / 2)
    '''

    xmin, ymin, b_width, b_height = filtered_boxes[i]
    xmin = int((xmin + 1) * width / 2)
    ymin = int((ymin + 1) * width / 2)
    b_width = int((b_width + 1) * height / 2)
    b_height = int((b_height + 1) * height / 2)

    #plot_bounding_box(image_path, xmin, ymin, xmin + b_width, ymin + b_height)

    print(f"Detection {i + 1}:")
    print(f"  Class: {filtered_class_ids[i]}")
    print(f"  Confidence: {filtered_confidences[i]:.2f}")
    #print(f"  Bounding box (pixel coordinates): [{xmin}, {ymin}, {xmax}, {ymax}]")
    print(f"  Bounding box (pixel coordinates): [{xmin}, {ymin}, {b_width}, {b_height}]")

    bboxes.append([xmin, ymin, b_width, b_height])

    # Draw rectangle on the image
    #cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    #cv2.rectangle(original_image, (xmin, ymin), (b_width, b_height), (0, 255, 0), 2)

    # Put class label and confidence score
    #label = f"Class {filtered_class_ids[i]}: {filtered_confidences[i]:.2f}"
    #cv2.putText(original_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Save the image with bounding boxes
#cv2.imwrite('output_image.jpg', original_image)
#print("Output image saved as 'output_image.jpg'")

plot_bounding_boxes(original_image, bboxes, filtered_confidences, filtered_class_ids)
