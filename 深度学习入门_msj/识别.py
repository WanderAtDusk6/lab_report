# -*- coding: utf-8 -*-
# 识别
import cv2
import numpy as np
from keras.models import load_model

model = load_model('my_model.h5')

image = cv2.imread('7.bmp.', 0)
img = cv2.imread('7.bmp', 0)

img = (img.reshape(1,28,28,1)).astype("float32")/255
predict = model.predict_classes(img)
print('识别为：')
print(predict)

cv2.imshow("Image1", image)
cv2.waitKey(0)
