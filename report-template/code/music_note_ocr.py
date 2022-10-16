# -*- coding: utf-8 -*-

import cv2
from keras import backend
from keras.callbacks import TensorBoard
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import load_model, save_model, Sequential
from keras.utils import np_utils
import numpy as np
from os import getcwd, listdir, path, remove
from os.path import join, isfile
from tempfile import mkstemp


def build_dataset():
    """
    Build a dataset, from existing dataset.
    """
    dataset = np.array([]);
    label = np.array([]);
    Len = 0
    for index in range(1, 10):
        rootdir = getcwd() + '\\dataset\\' + str(index)
        list = listdir(rootdir) 
        Len += len(list)
        for i in range(len(list)):
            path = join(rootdir, list[i])
            if isfile(path):
                # Gray-scale image.
                src = cv2.imread(path, 0)
                
                ret, src = cv2.threshold(src, 127, 255, 0)
                kernel = np.array([[1],[1],[1]], dtype = 'uint8') 
                src = cv2.dilate(src,kernel,iterations = 1)
                
                dataset = np.append(dataset, src)
                label = np.append(label, index - 1) 

    dataset = dataset.reshape(Len, 64, 32)

    index = np.arange(Len)
    np.random.shuffle(index)
    
    dataset = dataset[index, :, :]
    label = label[index]
    
    trainDataset = dataset[:-100]
    trainLabel = label[:-100]
    testDataset = dataset[-100:]
    testLabel = label[-100:]

    return testDataset, testLabel, trainDataset, trainLabel

def cnn_model(testDataset, testLabel, trainDataset, trainLabel):
    """
    CNN model.
    """
    batch_size = 16
    nb_classes = 9 
    nb_epoch = 8
    
    img_rows, img_cols = 64, 32 
    # Number of filters.
    nb_filters = 32
    # Convolutional Kernel size.
    kernel_size = (4, 4)
    # Pooling kernel size.
    pool_size = (2, 2)
    
    # Load data.
    (X_train, y_train), (X_test, y_test) = (trainDataset, trainLabel), (testDataset, testLabel)

    if backend.image_dim_ordering() == 'th':
        # Theano: (conv_dim1, channels, conv_dim2, conv_dim3).
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        # TensorFlow: (conv_dim1, conv_dim2, conv_dim3, channels).
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    # Binary matrix.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    """
    32 convolutional layer, one max pooling layer, and two FC layers.
    Use dropout to prevent from overfitting.
    Use RELU as activation function.
    Use Softmax as Cost function.
    """
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    # Compile the model.
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # Tensorboard.
    tbCallBack = TensorBoard(log_dir='.\\logs', 
        histogram_freq=0, 
        write_graph=True, 
        write_grads=True, 
        write_images=True, 
        embeddings_freq=0,
        embeddings_layer_names=None, 
        embeddings_metadata=None)
    
    # Train the model.
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test), callbacks=[tbCallBack])
    
    # Evaluate the model.
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0]) 
    print('Test accuracy:', score[1]) 
    
    # Save the model parameters.
    _, fname = mkstemp('.h5', dir='.\\save')
    save_model(model, fname)

if __name__ == '__main__':
    # Clear the logs.
    for file in listdir(getcwd() + '\\logs'):
        remove(getcwd() + '\\logs\\' + file)
    
    # Build dataset.
    testDataset, testLabel, trainDataset, trainLabel = build_dataset()
    # Train the model.
    cnn_model(testDataset, testLabel, trainDataset, trainLabel)
