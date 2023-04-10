import os
import math
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
tf.compat.v1.disable_eager_execution()


model = load_model('./models/thorax_with_background_gray_EfficientNetV2B0/unfreezed-model-0.953-460_FINAL.h5')

paths = [
    "./dataset/gray/thorax_with_background/all_eval",
    "./dataset/gray/thorax_with_background/od_predicted_all_eval"
]

for path in paths:
    print("Printing for Dataset: {}".format(path))

    datagen = ImageDataGenerator(rescale=1.0/255.0)
    batch_size, target_size, class_mode = 8, (224, 224), 'binary'

    print("Reading data ...")
    val_it = datagen.flow_from_directory(
        path + '/',
        class_mode=class_mode,
        batch_size=batch_size,
        target_size=target_size,
        shuffle=False)
    filenames = val_it.filenames

    print(val_it.class_indices)

    print("Predicting data ...")
    Y_pred = model.predict(val_it)
    y_pred = [(0, pred[0]) if pred[0] > pred[1] else (1, pred[1]) for pred in Y_pred]

    # print(val_it.labels)
    # print(y_pred)

    print("Calculating performance ...")
    confusion_matrix = {
        "real_non_rusty":   { "pred_non_rusty": 0, "pred_rusty": 0},
        "real_rusty":       { "pred_non_rusty": 0, "pred_rusty": 0}
    }

    for ind, filename in enumerate(filenames):

        # "non_rusty" ==> index 0
        if y_pred[ind][0] == 0:
            # predicted "non_rusty"
            predicted_class = "non_rusty {}%".format(round(y_pred[ind][1]*100), 2)

            if val_it.labels[ind] == 0:
                confusion_matrix["real_non_rusty"]["pred_non_rusty"] += 1
            else:
                confusion_matrix["real_rusty"]["pred_non_rusty"] += 1

        # "rusty" ==> index 1
        else:
            # predicted "rusty"
            predicted_class = "rusty {}%".format(round(y_pred[ind][1]*100), 2)

            if val_it.labels[ind] == 0:
                confusion_matrix["real_non_rusty"]["pred_rusty"] += 1
            else:
                confusion_matrix["real_rusty"]["pred_rusty"] += 1

    # # pos = non_rusty
    # # neg = rusty
    # TP = confusion_matrix["real_non_rusty"]["pred_non_rusty"]
    # TN = confusion_matrix["real_rusty"]["pred_rusty"]
    # FP = confusion_matrix["real_rusty"]["pred_non_rusty"]
    # FN = confusion_matrix["real_non_rusty"]["pred_rusty"]

    # pos = rusty
    # neg = non_rusty
    TP = confusion_matrix["real_rusty"]["pred_rusty"]
    TN = confusion_matrix["real_non_rusty"]["pred_non_rusty"]
    FP = confusion_matrix["real_non_rusty"]["pred_rusty"]
    FN = confusion_matrix["real_rusty"]["pred_non_rusty"]

    print("\nModel Performance\n")
    print("True Positive: ", TP)
    print("True Negative: ", TN)
    print("False Positive: ", FP)
    print("False Negative: ", FN)

    print()
    print("Accuracy: {:.2f}%".format((100*(TP+TN)) / (TP+TN+FP+FN)))
    print("F-1 Score: {:.2f}%".format(100*2*TP / ((2*TP) + FP + FN)))
    print("Precision: {:.2f}%".format(100 * TP / (TP + FP)))
    print("Sensitivity: {:.2f}%".format(100 * TP / (TP + FN)))
    print("Specificity: {:.2f}%".format(100 * TN / (FP + TN)))
    print("MACC: {:.2f}%".format( 100 * ((TP*TN) - (FP*FN)) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))))
