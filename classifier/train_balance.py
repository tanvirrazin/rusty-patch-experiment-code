import os
import sys
from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, GlobalMaxPooling2D, ZeroPadding2D, Convolution2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math, datetime, time
from tensorflow.keras.models import load_model
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

tf.compat.v1.disable_eager_execution()


datagen = ImageDataGenerator(rescale=1.0/255.0)

batch_size, target_size, class_mode = 8, (224, 224), 'binary'
number_of_epochs = 50
# prepare iterators
path = "./dataset/gray/thorax_without_background/"

train_it = datagen.flow_from_directory(os.path.join(path, 'train/'),
	batch_size=batch_size, target_size=target_size)
val_it = datagen.flow_from_directory(os.path.join(path, 'val/'),
	batch_size=batch_size, target_size=target_size)


base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# print(base_model.summary())

base_model.trainable = False
try:
  for layer in base_model.layers:
      layer.trainable = False
except: print("Freezing")

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)
base_model = Model(inputs=base_model.input, outputs=predictions)
base_model.compile(
	optimizer=Adam(learning_rate=0.0001),
	loss='binary_crossentropy',
	metrics=['accuracy'])

checkpoint = ModelCheckpoint(
        "models/thorax_without_background_gray_EfficientNetV2B0/freezed-model-{val_accuracy:.3f}-{epoch:03d}.h5",
        monitor='val_accuracy',
        verbose=1,
        save_best_only=False,
        mode='max',
        period=5
)

print(base_model.summary())
print("Train data size: ", len(train_it))
print("Val data size: ", len(val_it))

history = base_model.fit(
        train_it,
        steps_per_epoch=len(train_it),
        validation_data=val_it,
        validation_steps=len(val_it),
        epochs=number_of_epochs,
        callbacks=[checkpoint],
        verbose=1)

base_model.save('models/thorax_without_background_gray_EfficientNetV2B0/model_freezed.h5')

print(history.history)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.savefig('models/thorax_without_background_gray_EfficientNetV2B0/accuracies_freezed.png')
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('models/thorax_without_background_gray_EfficientNetV2B0/loss_freezed.png')
