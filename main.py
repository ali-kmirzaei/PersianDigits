import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential #FeedForward
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import os
import tensorflow as tf
import random

# =================================================================================================== #
# Prepare dataset:
SIZE = 150
tmp = list()
X_train = list()
y_train = list()
X_test = list()
y_test = list()
name_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

for name in name_list:
  path = 'dataset/'+name
  for item in os.listdir(path):
      img = plt.imread(os.path.join(path, item))
      img = cv.resize(img, (SIZE, SIZE))
      tmp.append((img, int(name)))

random.shuffle(tmp)

split = 70
for i in range(0, split):
  X_train.append(tmp[i][0])
  y_train.append(tmp[i][1])
for i in range(split, 100):
  X_test.append(tmp[i][0])
  y_test.append(tmp[i][1])

# Reshape data:
X_test = np.array(X_test)
X_train = np.array(X_train)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# # Validate data:
# for i in range(23, 32):
#     img = tmp[i][0]
#     y = tmp[i][1]
#     print(y)
#     while cv.waitKey(1) != ord('0'):
#         cv.imshow("Rotated", img)

# =================================================================================================== #
# Model:
EPOCHES = 50
LEARNING_RATE = 0.001
METRICS = ['accuracy']
LOSS = 'categorical_crossentropy'
VALIDATION_DATA = (X_test, y_test)


model = Sequential()
model.add(Input(shape=(SIZE, SIZE, 4)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10 , activation='softmax'))

opt = SGD(learning_rate = LEARNING_RATE)
model.compile(optimizer=opt, loss=LOSS, metrics=METRICS)
hist = model.fit(X_train, y_train, epochs=EPOCHES, validation_data=VALIDATION_DATA)

# =================================================================================================== #
# PLOT:
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('loss & Accuracy')
ax1.plot(hist.history['loss'] , color="orange" , label="train loss")
ax1.plot(hist.history['val_loss'] , color="blue" , label="validation loss")
ax1.legend()
ax2.plot(hist.history['accuracy'] , color="orange" , label="train accuracy")
ax2.plot(hist.history['val_accuracy'] , color="blue" , label="validation accuracy")
ax2.legend()
plt.show()
