import sys
import numpy as np
import tensorflow as tf

from preprocess import preprocess_image
from preprocess import AGE_CLASSES

model = tf.keras.models.load_model(
    "models/age_model.keras"
)

image_path = sys.argv[1]

image = preprocess_image(image_path)

prediction = model.predict(image)

index = np.argmax(prediction)

age = AGE_CLASSES[index]

confidence = np.max(prediction) * 100

print("\nРезультат:")

print(f"Возраст: {age}")

print(f"Уверенность: {confidence:.2f}%")
