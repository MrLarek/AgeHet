import tkinter as tk
import numpy as np
import tensorflow as tf

from tkinter import filedialog
from PIL import Image
from PIL import ImageTk

from preprocess import preprocess_image
from preprocess import AGE_CLASSES

model = tf.keras.models.load_model(
    "models/age_model.keras"
)

window = tk.Tk()

window.title("Age Recognizer")

window.geometry("500x600")

image_label = tk.Label(window)

image_label.pack(pady=10)

result_label = tk.Label(
    window,
    text="Загрузите фотографию",
    font=("Arial", 20)
)

result_label.pack(pady=10)


current_path = None


def load_image():

    global current_path

    path = filedialog.askopenfilename(
        filetypes=[
            ("Изображения", "*.jpg *.jpeg *.png")
        ]
    )

    if not path:
        return

    current_path = path

    img = Image.open(path)
    img = img.resize((300, 300))

    photo = ImageTk.PhotoImage(img)

    image_label.config(image=photo)
    image_label.image = photo

    result_label.config(
        text="Нажмите Распознать"
    )


def predict_age():

    if current_path is None:
        return

    image = preprocess_image(current_path)

    prediction = model.predict(
        image,
        verbose=0
    )

    index = np.argmax(prediction)

    age = AGE_CLASSES[index]

    confidence = np.max(prediction) * 100

    result_label.config(
        text=f"Возраст: {age} ({confidence:.2f}%)"
    )


load_button = tk.Button(
    window,
    text="Загрузить фото",
    command=load_image,
    font=("Arial", 14)
)

load_button.pack(pady=5)

predict_button = tk.Button(
    window,
    text="Распознать",
    command=predict_age,
    font=("Arial", 14)
)

predict_button.pack(pady=5)

window.mainloop()
