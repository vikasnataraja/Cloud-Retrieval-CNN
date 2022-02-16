from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Activation, Dropout
from keras.layers import Concatenate, Input, LeakyReLU
from keras.initializers import RandomNormal
from keras.models import Model
from utils.model_utils import BatchNorm, UpSample
import keras.backend as K

kernel_init = RandomNormal(stddev=0.02)

def upsample_transpose(layer, filters, kernel_size, strides=(1, 1),  pad_type='same', method='nearest'):
    """ Upsampling block with interpolation and transposed convolution """
    x = UpSampling2D(size=(2, 2), interpolation=method)(layer)
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                        strides=strides, padding=pad_type)(x)
    x = BatchNorm()(x)
    x = Activation('relu')(x)
    return x

"""
def conv_block(layer, filters, kernel_size, strides=(1, 1), pad_type='same', decoder=False, dropout=False):
    
    # Convolutional Block consisting of Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=pad_type, kernel_initializer=kernel_init)(layer)
    x = BatchNorm()(x)
    x = Activation('relu')(x) if decoder else LeakyReLU(alpha=0.2)(x)
    if dropout:
        x = Dropout(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=pad_type, kernel_initializer=kernel_init)(x)
    x = BatchNorm()(x)
    x = Activation('relu')(x) if decoder else LeakyReLU(alpha=0.2)(x)
    if dropout:
        x = Dropout(0.2)(x)
    return x

"""
def conv_block(layer, filters, kernel_size, strides=(1, 1), pad_type='same'):
    # Convolutional Block consisting of Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, padding=pad_type)(layer)
    x = BatchNorm()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size,
                strides=strides, padding=pad_type)(x)
    x = BatchNorm()(x)
    x = Activation('relu')(x)
    return x


def unet(input_shape, num_channels, num_classes, final_activation_fn):
    """ Build UNet """
    input_layer = Input((input_shape, input_shape, num_channels))

    conv1 = conv_block(layer=input_layer, filters=64, kernel_size=(3, 3), strides=(1, 1), pad_type='same')
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = conv_block(layer=pool1, filters=128, kernel_size=(3, 3), strides=(1, 1), pad_type='same')
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    conv3 = conv_block(layer=pool2, filters=256, kernel_size=(3, 3), strides=(1, 1), pad_type='same')
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

    conv4 = conv_block(layer=pool3, filters=512, kernel_size=(3, 3), strides=(1, 1), pad_type='same')
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)

    conv5 = conv_block(layer=pool4, filters=1024, kernel_size=(3, 3), strides=(1, 1), pad_type='same')
    up6 = UpSampling2D(size=(2, 2), interpolation='nearest')(conv5)
    # up6 = upsample_transpose(layer=conv5, filters=512, kernel_size=(2, 2), strides=(1, 1), pad_type='same')
    # up6 = Dropout(0.5)(up6)
    concat6 = Concatenate(axis=-1)([conv4, up6])

    conv6 = conv_block(layer=concat6, filters=512, kernel_size=(3, 3), strides=(1, 1), pad_type='same')
    up7 = UpSampling2D(size=(2, 2), interpolation='nearest')(conv6)
    # up7 = upsample_transpose(layer=conv6, filters=256, kernel_size=(2, 2), strides=(1, 1), pad_type='same')
    # up7 = Dropout(0.5)(up7)
    concat7 = Concatenate(axis=-1)([conv3, up7])

    conv7 = conv_block(layer=concat7, filters=256, kernel_size=(3, 3), strides=(1, 1), pad_type='same')
    up8 = UpSampling2D(size=(2, 2), interpolation='nearest')(conv7)
    # up8 = upsample_transpose(layer=conv7, filters=128, kernel_size=(2, 2), strides=(1, 1), pad_type='same')
    # up8 = Dropout(0.5)(up8)
    concat8 = Concatenate(axis=-1)([conv2, up8])

    conv8 = conv_block(layer=concat8, filters=128, kernel_size=(3, 3), strides=(1, 1), pad_type='same')
    up9 = UpSampling2D(size=(2, 2), interpolation='nearest')(conv8)
    # up9 = upsample_transpose(layer=conv8, filters=64, kernel_size=(2, 2), strides=(1, 1), pad_type='same')
    # up9 = Dropout(0.5)(up9)
    concat9 = Concatenate(axis=-1)([conv1, up9])

    conv9 = conv_block(layer=concat9, filters=64, kernel_size=(3, 3), strides=(1, 1), pad_type='same')
    conv10 = Conv2D(filters=num_classes, kernel_size=(1, 1))(conv9)
    conv10 = Activation(final_activation_fn)(conv10)

    model = Model(inputs=input_layer, outputs=conv10)
    return model


