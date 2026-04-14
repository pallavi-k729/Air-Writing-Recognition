import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 100
DATASET_PATH = "../dataset"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.3,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

labels = train_gen.class_indices
np.save("../label_maps/coarse_labels.npy", labels)

inp = Input(shape=(64,64,1))
x = Conv2D(32,(3,3),activation='relu',padding='same')(inp)
x = MaxPooling2D()(x)
x = Dropout(0.3)(x)
x = Conv2D(64,(3,3),activation='relu',padding='same')(x)
x = MaxPooling2D()(x)
x = Dropout(0.4)(x)
x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
x = MaxPooling2D()(x)
x = Dropout(0.4)(x)
x = Flatten()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(len(labels),activation='softmax')(x)

model = Model(inp,out)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[lr]
)
model.save("coarse_model.h5")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.title('Coarse Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
###
plt.subplot(1,2,2)
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.title('Coarse Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("coarse_plot.png")
plt.show()