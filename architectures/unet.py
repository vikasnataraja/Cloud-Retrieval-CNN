from keras.layers import MaxPooling2D, ZeroPadding2D, UpSampling2D, Cropping2D
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, Dropout
from keras.layers import Concatenate, Input
from keras.models import Model
from utils.model_utils import BatchNorm, UpSample

def UpSampleTranspose(layer, filters, kernel_size, pad_type, method='bilinear'):
  x = UpSampling2D(size=(2,2), interpolation=method)(layer)
  x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding=pad_type)(x)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  return x

def UpSampleConv(layer, filters, kernel_size, pad_type, method='bilinear'):
  x = UpSampling2D(size=(2,2), interpolation=method)(layer)
  x = Conv2D(filters=filters, kernel_size=kernel_size, padding=pad_type)(x)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  return x

def ConvBlock(layer, filters, kernel_size, pad_type):
  x = Conv2D(filters=filters, kernel_size=kernel_size, padding=pad_type)(layer)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  x = Conv2D(filters=filters, kernel_size=kernel_size, padding=pad_type)(x)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  return x
  
def UNet(input_shape, num_channels, num_classes, final_activation_fn):
  input_layer = Input((input_shape,input_shape,num_channels))
  
  conv1 = ConvBlock(layer=input_layer, filters=64, kernel_size=(3,3), pad_type='same')
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = ConvBlock(layer=pool1, filters=128, kernel_size=(3,3), pad_type='same')
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  
  conv3 = ConvBlock(layer=pool2, filters=256, kernel_size=(3,3), pad_type='same')
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  
  conv4 = ConvBlock(layer=pool3, filters=512, kernel_size=(3,3), pad_type='same')
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

  conv5 = ConvBlock(layer=pool4, filters=1024, kernel_size=(3,3), pad_type='same')
  up6 = UpSampleTranspose(layer=conv5, filters=512, kernel_size=(2,2), pad_type='same')
  concat6 = Concatenate(axis=-1)([conv4, up6])

  conv6 = ConvBlock(layer=concat6, filters=512, kernel_size=(3,3), pad_type='same')
  up7 = UpSampleTranspose(layer=conv6, filters=256, kernel_size=(2,2), pad_type='same')
  concat7 = Concatenate(axis=-1)([conv3, up7])

  conv7 = ConvBlock(layer=concat7, filters=256, kernel_size=(3,3), pad_type='same')
  up8 = UpSampleTranspose(layer=conv7, filters=128, kernel_size=(2,2), pad_type='same')
  concat8 = Concatenate(axis=-1)([conv2,up8])

  conv8 = ConvBlock(layer=concat8, filters=128, kernel_size=(3,3), pad_type='same')
  up9 = UpSampleTranspose(layer=conv8, filters=64, kernel_size=(2,2), pad_type='same')
  concat9 = Concatenate(axis=-1)([conv1,up9])

  conv9 = ConvBlock(layer=concat9, filters=64, kernel_size=(3,3), pad_type='same')
  conv10 = Conv2D(num_classes, kernel_size=(1,1))(conv9)
  conv10 = Activation(final_activation_fn)(conv10)
  
  model = Model(inputs=input_layer, outputs=conv10)
  return model
