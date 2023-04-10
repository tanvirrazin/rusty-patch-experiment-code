import os
import time
import pathlib
import tensorflow as tf
import cv2
import pprint
import argparse

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

# # Enable GPU dynamic memory allocation
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)



# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = './exported_models/rusty_thorax_efficientdetD1_2023_01_31_steps_100000'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = './annotations/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.20)

# LOAD THE MODEL
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)



def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))



def process_image(image_path):

    print('Running inference for {}... '.format(image_path), end='')

    image_path_parts = image_path.rsplit('.', 1)
    predicted_image_path = image_path_parts[0] + '_thorax_predicted.' + image_path_parts[1]
    cropped_image_path = image_path_parts[0] + '_thorax_cropped.' + image_path_parts[1]

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # pprint.pprint(detections['detection_boxes'][:3])

    image_with_detections = image.copy()

    for i_detection in range(5):
        if detections['detection_scores'][i_detection] >= MIN_CONF_THRESH:
            ymin, xmin, ymax, xmax = detections['detection_boxes'][i_detection]
            im_height, im_width = image.shape[:2]
            (xminn, xmaxx, yminn, ymaxx) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
            cropped_image = image[yminn:ymaxx, xminn:xmaxx, :]
            cropped_image_path = image_path_parts[0] + '_thorax_cropped_' + str(i_detection) + '.' + image_path_parts[1]
            cv2.imwrite(cropped_image_path, cropped_image)

    # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=0.2,
          agnostic_mode=False)

    print('Done')

    cv2.imwrite(predicted_image_path, image_with_detections)

if __name__ == "__main__":
    # PROVIDE PATH TO IMAGE DIRECTORY

    image_paths = [
        "./test_results"
    ]
    for path in image_paths:
        for filename in os.listdir(path):
            process_image(os.path.join(path, filename))
