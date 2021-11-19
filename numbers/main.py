
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import glob
#gpu acceleration
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#dataset: https://www.kaggle.com/xainano/handwrittenmathsymbols

#prepping train directory
read_path = "ds/data/extracted_images/"
#train variables
x_train = []
x_valid = []
#validation variables
y_train = []
y_valid = []
#testing variables
x_test = []
y_test = []
#adding subdirectories(symbols) to train
folders = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', '=', 'leq', 'neq', 'geq', 'alpha', 'beta', 'lambda', 'lt', 'gt', 'x', 'y']

for i in range(len(folders)):
    xfiles = glob.glob(read_path + folders[i] + "/*.jpg")
    valid_counter = 150
    test_counter = 50
    for img in xfiles:
        image = img_to_array(load_img(img))
        if valid_counter > 0:
            x_valid.append(image)
            y_valid.append(np.array(i))
            valid_counter -= 1
        elif test_counter > 0:
            x_test.append(image)
            y_test.append(np.array(i))
            test_counter -= 1
        else:
            x_train.append(image)
            y_train.append(np.array(i))

#data has to be numpy array
x_train = np.array(x_train)
x_valid = np.array(x_valid)
x_test = np.array(x_test)


#one-hot encode target column
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
y_test = to_categorical(y_test)
y_train = np.array(y_train)
y_valid = np.array(y_valid)
y_test = np.array(y_test)
print(y_train.shape)
#check number of categories
print(y_train[0].shape)

#create model
model = keras.Sequential()
#base of model (conv -> pool -> conv -> pool)
model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=[45,45,3]))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#head of model
model.add(layers.Flatten())
model.add(layers.Dense(23, activation='softmax'))
#compile model for categories
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

#fitting the model
history = model.fit(x_train, y_train,
          epochs=6,
          validation_data=(x_valid, y_valid))

model.save('numbers_model')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
