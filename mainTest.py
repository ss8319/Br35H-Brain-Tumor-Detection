import cv2 
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image


model = load_model("BrainTumor10Epochs.h5")

#2 Models to choose from
image = cv2.imread("D:\Br35H  Brain Tumor Detection 2020\pred\pred56.jpg")
img = Image.fromarray(image)
img = img.resize((64,64))
img = np.array(img) 
input_img = np.expand_dims(img, axis=0)

prob = model.predict(input_img)
yhat = (prob >= 0.5).astype(int)
print(f"decisions = \n{yhat}")
# # Get class labels
# class_label = np.argmax(prob, axis=-1)
# print(class_label)

# Output of 1: Tumor
# Output of 0: No Tumor