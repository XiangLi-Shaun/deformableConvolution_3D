import keras.backend as K
from keras.optimizers import *
from keras.models import Model
from keras.layers import concatenate, add, Activation, BatchNormalization, core,Dropout,Input, Dense, merge, Conv3D, UpSampling3D, Flatten, Reshape
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Lambda, regularizers, Flatten
from keras.activations import softmax

from dcn_3d.layer import DCNN3D

def get_net(nb_batch, patch_z, patch_height, patch_width,n_ch):
    inputs = Input((patch_z, patch_height, patch_width, n_ch))

    # conv1 = Conv3D(8, (3, 3, 3), activation='relu', padding='valid')(inputs)
    conv1 = DCNN3D(nb_batch, 1, (3, 3, 3), scope='deformconv1',norm=False)(inputs)

    # conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='valid')(conv1)
    conv2 = DCNN3D(nb_batch, 32, (3, 3, 3), scope='deformconv2')(conv1)

    conv2 = BatchNormalization(axis=-1)(conv2)

    pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)

    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='valid')(pool2)


    conv2 = BatchNormalization(axis=-1)(conv2)

    pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)



    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='valid')(pool2)

    conv2 = BatchNormalization(axis=-1)(conv2)

    pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)

    pool3 = Flatten()(pool2)
    fc1 = Dense(128, activation='relu')(pool3)
    fc1 = Dropout(0.2)(fc1)

    out = Dense(10, activation='relu')(fc1)
    out = Lambda(lambda out: softmax(out, axis=-1))(out)
    # model
    model = Model(inputs=[inputs], outputs=[out])

    model.compile(optimizer='Adadelta', loss='categorical_crossentropy',metrics=['categorical_crossentropy', 'accuracy'])


    return model