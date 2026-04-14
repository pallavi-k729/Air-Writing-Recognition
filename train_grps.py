import os
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
EPOCHS = 20
BASE_DATASET = "../dataset"
BASE_SAVE = "../fine_models"
LABEL_PATH = "../label_maps"
groups = sorted(os.listdir(BASE_DATASET))

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1],1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(input_shape[1],1), initializer='zeros', trainable=True)
    def call(self, x):
        e = tf.matmul(x,self.W)+self.b
        e = tf.nn.tanh(e)
        a = tf.nn.softmax(e,axis=1)
        return tf.reduce_sum(x*a,axis=1)

for group in groups:
    dataset_path = os.path.join(BASE_DATASET, group)
    save_path = os.path.join(BASE_SAVE, group)
    os.makedirs(save_path, exist_ok=True)

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.3,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2
    )

    train_gen = datagen.flow_from_directory(dataset_path, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, color_mode="grayscale", class_mode="categorical", subset="training")
    val_gen = datagen.flow_from_directory(dataset_path, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, color_mode="grayscale", class_mode="categorical", subset="validation")
    labels = train_gen.class_indices
    np.save(os.path.join(LABEL_PATH, f"{group}_labels.npy"), labels)

    inp = Input(shape=(64,64,1))
    x = Conv2D(32,(3,3),activation='relu',padding='same')(inp)
    x = MaxPooling2D()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.4)(x)
    x = Reshape((-1,x.shape[1]*x.shape[3]))(x)
    x = Bidirectional(LSTM(32,return_sequences=True))(x)
    x = Bidirectional(LSTM(32,return_sequences=True))(x)
    x = AttentionLayer()(x)
    x = Dense(128,activation='relu')(x)
    out = Dense(len(labels),activation='softmax')(x)

    model = Model(inp,out)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3)
    history = model.fit(train_gen,validation_data=val_gen,epochs=EPOCHS,callbacks=[lr])
    model.save(os.path.join(save_path, "model.h5"))

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'],label='Train Accuracy')
    plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
    plt.title(f'{group} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'],label='Train Loss')
    plt.plot(history.history['val_loss'],label='Validation Loss')
    plt.title(f'{group} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "plot.png"))
    plt.close()
    print(f"Finished training thee {group}")