import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocess import IMG_SIZE
from preprocess import AGE_CLASSES

DATASET_DIR = "datasets"
CSV_PATH = "datasets/age_detection.csv"


def load_data(split):

    data = pd.read_csv(CSV_PATH)

    data = data[data["split"] == split]

    images = []
    labels = []

    for _, row in data.iterrows():

        path = os.path.join(DATASET_DIR, row["file"])

        image = cv2.imread(path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

        images.append(resized)

        labels.append(AGE_CLASSES.index(row["age"]))

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


x_train, y_train = load_data("train")
x_test, y_test = load_data("test")

x_train = x_train / 255.0
x_test = x_test / 255.0

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2)
)

datagen.fit(x_train)

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
)

model.add(MaxPooling2D((2, 2)))

model.add(
    Conv2D(
        64,
        (3, 3),
        activation='relu'
    )
)

model.add(MaxPooling2D((2, 2)))

model.add(
    Conv2D(
        128,
        (3, 3),
        activation='relu'
    )
)

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(
    Dense(
        256,
        activation='relu'
    )
)

model.add(
    Dropout(0.4)
)

model.add(
    Dense(
        len(AGE_CLASSES),
        activation='softmax'
    )
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    datagen.flow(x_train, y_train, batch_size=16),
    epochs=50,
    validation_data=(x_test, y_test)
)

model.save("models/age_model.keras")

print("Модель сохранена.")
