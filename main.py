import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


# image = cv2.imread('image_part_003.jpg')
# image = np.resize(image, (256, 256, 3))
# image = image.reshape(-1, 256, 256, 3)
model = tf.keras.saving.load_model('Mymodel_1st_try.h5')
# pre = model.predict(image)[0].argmax(axis=2) + 1
# plt.imshow(image)
# plt.show()

def image_mod(image):
    image = np.resize(image,(256,256,3))
    image = image.reshape(-1,256,256,3)
    return model.predict(image)[0].argmax(axis=2) + 1


demo = gr.Interface(
    image_mod,
    gr.Image(type="numpy"),
    "image",
    flagging_options=["blurry", "incorrect", "other"],

)

if __name__ == "__main__":
    demo.launch()