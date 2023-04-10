import sys
import json
from matplotlib import pyplot
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras import optimizers, Sequential
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


datagen = ImageDataGenerator(rescale=1.0/255.0)

batch_size, target_size, class_mode = 8, (224, 224), 'binary'
number_of_epochs = 1000
# prepare iterators
path = "./dataset/gray/thorax_without_background/"

train_it = datagen.flow_from_directory(path + 'train/',
	batch_size=batch_size, target_size=target_size)
val_it = datagen.flow_from_directory(path + 'val/',
	batch_size=batch_size, target_size=target_size)

model = load_model('models/thorax_without_background_gray_EfficientNetV2B0/freezed-model-0.500-050.h5')

# Unfreeze model
model.trainable = True
try:
  for layer in model.layers:
    layer.trainable = True
except: print('Unfreezing')


model.compile(
	optimizer=Adam(learning_rate=0.0001),
	loss='binary_crossentropy',
	metrics=['accuracy'])

checkpoint = ModelCheckpoint(
        "models/thorax_without_background_gray_EfficientNetV2B0/unfreezed-model-{val_accuracy:.3f}-{epoch:02d}.h5",
        monitor='val_accuracy',
        verbose=1,
        save_best_only=False,
        mode='max',
        period=5
)

history = model.fit(
	train_it,
	steps_per_epoch=len(train_it),
	validation_data=val_it,
	validation_steps=len(val_it),
	epochs=number_of_epochs,
        callbacks=[checkpoint],
	verbose=1)

model.save('models/thorax_without_background_gray_EfficientNetV2B0/model_unfreezed.h5')

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
plt.savefig('models/thorax_without_background_gray_EfficientNetV2B0/accuracies_unfreezed.png')
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('models/thorax_without_background_gray_EfficientNetV2B0/loss_unfreezed.png')

with open ("models/thorax_without_background_gray_EfficientNetV2B0/stat.json", "w") as statfile:
	json.dump({
		"train_acc": acc,
		"val_acc": val_acc,
		"train_loss": train_loss,
		"val_loss": val_loss
	}, statfile)
