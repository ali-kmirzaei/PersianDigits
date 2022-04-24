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


# gpu_devices = tf.config.experimental.list_physical_devices("GPU")
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

# =================================================================================================== #
# Prepare dataset:

SIZE = 150

def data_prepare():
    tmp = list()
    X_train = list()
    y_train = list()
    X_test = list()
    y_test = list()
    name_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for name in name_list:
      path = 'dataset/'+name
      for item in os.listdir(path):
          img = cv.imread(os.path.join(path, item)) / 255
          # img = plt.imread(os.path.join(path, item))
          # print(img.shape)
          img = cv.resize(img, (SIZE, SIZE))
          tmp.append((img, int(name)))

    random.shuffle(tmp)

    total = len(tmp)
    split = int(70 * total / 100)
    # print(total)
    # print(split)

    for i in range(0, split):
      X_train.append(tmp[i][0])
      y_train.append(tmp[i][1])
    for i in range(split, total):
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
    #     img = X_test[i]
    #     y = y_test[i]
    #     print(y)
    #     while cv.waitKey(1) != ord('0'):
    #         cv.imshow("Rotated", img)

    return X_test, X_train, y_train, y_test

# =================================================================================================== #
# Model:

def model(X_test, X_train, y_train, y_test):
    EPOCHES = 100
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    METRICS = ['accuracy']
    LOSS = 'categorical_crossentropy'
    VALIDATION_DATA = (X_test, y_test)

    model = Sequential()
    model.add(Input(shape=(SIZE, SIZE, 3)))
    model.add(Flatten())
    # model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.9))
    model.add(Dense(10 , activation='softmax'))

    opt = SGD(learning_rate = LEARNING_RATE)
    model.compile(optimizer=opt, loss=LOSS, metrics=METRICS)


    checkpoint_path = "cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=5)

    # model.load_weights(checkpoint_path)
    #
    # x = X_test[:1]
    # a = np.argmax(model.predict(x)[0])
    # print(a)
    # while cv.waitKey(1) != ord('0'):
    #     cv.imshow("Rotated", X_test[0])
    # hist = 0

    hist = model.fit(X_train, y_train,
                     epochs=EPOCHES, batch_size=BATCH_SIZE,
                     validation_data=VALIDATION_DATA
                     ,callbacks=[cp_callback])

    return model, hist

# =================================================================================================== #
# PLOT:
def plot(hist):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('loss & Accuracy')
    ax1.plot(hist.history['loss'] , color="orange" , label="train loss")
    ax1.plot(hist.history['val_loss'] , color="blue" , label="validation loss")
    ax1.legend()
    ax2.plot(hist.history['accuracy'] , color="orange" , label="train accuracy")
    ax2.plot(hist.history['val_accuracy'] , color="blue" , label="validation accuracy")
    ax2.legend()
    plt.show()

# =================================================================================================== #
# Calls:

X_test, X_train, y_train, y_test = data_prepare()



# X_train = list()
# y_train = list()
# X_test = list()
# y_test = list()
# path = 'test/'
# for item in os.listdir(path):
#     img = plt.imread(os.path.join(path, item))
#     img = cv.resize(img, (SIZE, SIZE))
#     X_test.append(img)
# X_test = np.array(X_test)
# print(X_test.shape)



model, hist = model(X_test, X_train, y_train, y_test)
plot(hist)













# END
