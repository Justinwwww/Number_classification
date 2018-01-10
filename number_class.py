import numpy as np
np.random.seed(123)
from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.datasets import mnist

#Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()


#Preprocessing
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print Y_train.shape

#Build Model
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28),
                        dim_ordering='th'))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.48))
model.add(Dense(10, activation='sigmoid'))
#Compile Model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#Fit Model
model.fit(X_train, Y_train,
          batch_size=32, nb_epoch=2, verbose=1)
#Evaluate Model
score = model.evaluate(X_test, Y_test, verbose=0)