import matplotlib.pyplot as plt

from preprocess import preprocess_image

img = preprocess_image(
    "images/test.jpg"
)

plt.imshow(
    img[0]
)

plt.show()
