
# Set the random seed
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from keras.utils import normalize, print_summary, plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, Conv1D, AveragePooling1D



def CNN(num_classes, emotion='valence'):


    if emotion == 'valence':
        lr = 0.00001
        act_func = 'tanh'
    elif emotion == 'arousal':
        lr = 0.001
        act_func = 'relu'

    model = Sequential()

    model.add(Conv2D(100,  kernel_size=(3,3), padding='valid', input_shape=(1,40,101),
                     data_format='channels_first'))

    model.add(Activation(act_func))

    model.add(Conv2D(100,  kernel_size=(3,3)))

    model.add(Activation('tanh'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.50))

    model.add(Flatten())

    model.add(Dense(128))

    model.add(Activation('tanh'))

    model.add(Dropout(0.25))


    model.add(Dense(num_classes))


    model.add(Activation('softplus'))


    sgd = SGD(lr = lr, decay = 1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['categorical_accuracy'])

    return model



def DNN(num_classes):

    model = Sequential()

    # Input Dropout

    model.add(Dropout(0.25, batch_input_shape=(None, 4040)))

    # FC1
    model.add(Dense(5000, activation='relu'))
    model.add(Dropout(0.5))

    # FC2
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))

    # FC3
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))

    # FC4
    model.add(Dense(num_classes, activation='softmax'))

    rmsprop = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    #print_summary(model)


    return model


def Cnn1D(num_classes):


    model = Sequential()

    model.add(Conv1D(filters=16, kernel_size=5, bathc_input_shape=(50, 1,  4040),\
                        kernel_initializer='he_normal'))

    model.add(Activation('relu'))


    model.add(Conv1D(filters=32, kernel_size=5, kernel_initializer='he_normal'))

    model.add(Activation('relu'))


    model.add(Conv1D(filters=64, kernel_size=5, kernel_initializer='he_normal'))

    model.add(Activation('relu'))

    model.add(AveragePooling1D(pool_size=7))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    adam = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy',\
            metrics=['categorical_accuracy'])


    return model
