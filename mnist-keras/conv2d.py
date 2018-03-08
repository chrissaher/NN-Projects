import numpy as np
import keras
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
from keras.datasets import mnist


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    m, w, h = x_train.shape

    K.clear_session()
    sess = K.get_session()

    x = np.array(x_train).reshape(m, w, h, 1)
    y = sess.run(K.one_hot(y_train, 10))

    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(5,5), strides=(1,1), padding='valid', activation = 'relu', name = 'CONV1', input_shape=(w, h, 1)))
    model.add(layers.BatchNormalization(name='BN1'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=None, padding='valid', name="MP1"))

    model.add(layers.Conv2D(16, kernel_size=(3,3), strides=(1,1), padding='valid', activation = 'relu', name = 'CONV2'))
    model.add(layers.BatchNormalization(name='BN2'))
    model.add(layers.MaxPool2D(pool_size=(2,2), strides=None, padding='valid', name="MP2"))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation="softmax"))

    sgd = optimizers.SGD(lr = 0.001, momentum=0.9)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x, y, epochs=5, batch_size=64, verbose = True)


    x_test = np.array(x_test).reshape(tc, w, h, 1)
    y_test = sess.run(K.one_hot(y_test, 10))

    test_loss, test_acc = model_conv.evaluate(x_test, y_test)
    print(test_acc)
