import os, math, time, cv2, sys, shutil
import numpy as np
from os.path import isfile, join
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
# from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input
# from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
tf.compat.v1.disable_eager_execution()


MODEL_PATH = "./models/whole_body_gray_EfficientNetV2B0/unfreezed-model-0.931-500_FINAL.h5"

def execute(image_dir):
    predicted_name = ""
    predicted_probability = 0

    print("Loading model ....")
    model = load_model(MODEL_PATH)
    # print(model.summary())

    # class prediction
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    batch_size, target_size, class_mode = 8, (224, 224), 'binary'

    val_it = datagen.flow_from_directory(
        image_dir,
    	class_mode=class_mode,
        batch_size=batch_size,
        target_size=target_size,
        shuffle=False)
    filenames = val_it.filenames

    Y_pred = model.predict(val_it)
    y_pred = [(0, pred[0]) if pred[0] > pred[1] else (1, pred[1]) for pred in Y_pred]

    print(y_pred)

    for ind, filename in enumerate(filenames):

        # if y_pred[ind][0] == 0:
        #     predicted_class = "female {}%".format(round(y_pred[ind][1]*100), 2)
        # else:
        #     predicted_class = "male {}%".format(round(y_pred[ind][1]*100), 2)
        
        # print(predicted_class)

        if y_pred[ind][0] == val_it.labels[ind]:
            predicted_bool = "correct"
        else:
            predicted_bool = "wrong"

        print(filename, " -- ", predicted_bool)


        # CAM Generation

        img_path = os.path.join(image_dir + filename)
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        argmax = np.argmax(preds[0])
        output = model.output[:, argmax]

        # last_conv_layer = model.get_layer('conv5_block3_3_conv')
        last_conv_layer = model.get_layer('top_activation')
        # last_conv_layer = model.get_layer('conv5_block3_out')
        
        
        grads = K.gradients(output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])

        # for i in range(512):
        #     conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        hif = 0.8
        superimposed_img = heatmap * hif + img


        filename_parts = filename.split('.')
        predicted_probability = round(y_pred[ind][1]*100, 2)
        cv2.imwrite(os.path.join(image_dir, filename_parts[0] + "_" + predicted_bool + "_" + str(predicted_probability) + "_CAM." + filename_parts[1]), superimposed_img)






image_dir_paths = [
    # "/data/bhuiyan/bee/rusty/classifier/dataset/CAMs/val/",
    "/data/bhuiyan/bee/rusty/classifier/dataset/CAMs/try/"
]



print("Running prediction ....")
for image_dir_path in image_dir_paths:
    classes = execute(image_dir_path)





