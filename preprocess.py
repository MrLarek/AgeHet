import cv2
import numpy as np

IMG_SIZE = 128

AGE_CLASSES = [
    "18-20",
    "21-30",
    "31-40",
    "41-50",
    "51-60"
]


def preprocess_image(image_path):

    with open(image_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

    normalized = resized / 255.0

    reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    return reshaped