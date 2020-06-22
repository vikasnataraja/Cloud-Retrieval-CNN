from keras.layers import MaxPooling2D, ZeroPadding2D, UpSampling2D, Cropping2D
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, Dropout
from keras.layers import Concatenate, Add, Multiply, Input
from keras.models import Model
from utils.model_utils import BatchNorm, UpSample
import keras.backend as K


def UpSampleTranspose(layer, filters, kernel_size, strides=(1,1),  pad_type='same', method='bilinear'):
  x = UpSampling2D(size=(2,2), interpolation=method)(layer)
  x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=pad_type)(x)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  return x

def UpSampleConv(layer, filters, kernel_size, strides=(1,1), pad_type='same', method='bilinear'):
  x = UpSampling2D(size=(2,2), interpolation=method)(layer)
  x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=pad_type)(x)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  return x

def ConvBlock(layer, filters, kernel_size, strides=(1,1), pad_type='same'):
  x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=pad_type)(layer)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=pad_type)(x)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  return x
  
def UNet(input_shape, num_channels, num_classes, final_activation_fn):
  input_layer = Input((input_shape,input_shape,num_channels))
  
  conv1 = ConvBlock(layer=input_layer, filters=64, kernel_size=(3,3), strides=(1,1), pad_type='same')
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  
  conv2 = ConvBlock(layer=pool1, filters=128, kernel_size=(3,3), strides=(1,1), pad_type='same')
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  
  conv3 = ConvBlock(layer=pool2, filters=256, kernel_size=(3,3), strides=(1,1), pad_type='same')
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  
  conv4 = ConvBlock(layer=pool3, filters=512, kernel_size=(3,3), strides=(1,1), pad_type='same')
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

  conv5 = ConvBlock(layer=pool4, filters=1024, kernel_size=(3,3), strides=(1,1), pad_type='same')
  up6 = UpSampleTranspose(layer=conv5, filters=512, kernel_size=(2,2), strides=(1,1), pad_type='same')
  # up6 = Dropout(0.5)(up6)
  concat6 = Concatenate(axis=-1)([conv4, up6])

  conv6 = ConvBlock(layer=concat6, filters=512, kernel_size=(3,3), strides=(1,1), pad_type='same')
  up7 = UpSampleTranspose(layer=conv6, filters=256, kernel_size=(2,2), strides=(1,1), pad_type='same')
  # up7 = Dropout(0.5)(up7)
  concat7 = Concatenate(axis=-1)([conv3, up7])

  conv7 = ConvBlock(layer=concat7, filters=256, kernel_size=(3,3), strides=(1,1), pad_type='same')
  up8 = UpSampleTranspose(layer=conv7, filters=128, kernel_size=(2,2), strides=(1,1), pad_type='same')
  # up8 = Dropout(0.5)(up8)
  concat8 = Concatenate(axis=-1)([conv2,up8])

  conv8 = ConvBlock(layer=concat8, filters=128, kernel_size=(3,3), strides=(1,1), pad_type='same')
  up9 = UpSampleTranspose(layer=conv8, filters=64, kernel_size=(2,2), strides=(1,1), pad_type='same')
  # up9 = Dropout(0.5)(up9)
  concat9 = Concatenate(axis=-1)([conv1,up9])

  conv9 = ConvBlock(layer=concat9, filters=64, kernel_size=(3,3), strides=(1,1), pad_type='same')
  conv10 = Conv2D(num_classes, kernel_size=(1,1))(conv9)
  conv10 = Activation(final_activation_fn)(conv10)
  
  model = Model(inputs=input_layer, outputs=conv10)
  return model

def AttnGatingBlock(x, g, inter_shape):
  """ take g(spatially smaller signal) ->conv to get the same
  number of feature channels as x (bigger spatially)
  do a conv on x to also get same feature channels (theta_x)
  then, upsample g to be same size as x 
  add x and g (concat_xg)
  relu, 1x1 conv, then sigmoid/softmax then upsample the final - this gives us attn coefficients"""
    
  shape_x = K.int_shape(x)  
  shape_g = K.int_shape(g)  
  theta_x = Conv2D(inter_shape, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)  # 16
  shape_theta_x = K.int_shape(theta_x)
    
  phi_g = Conv2D(inter_shape, kernel_size=(1, 1), padding='same')(g)
  upsampled_g = Conv2DTranspose(inter_shape, kernel_size=(3, 3),
                                strides=(shape_theta_x[1]//shape_g[1], shape_theta_x[2]//shape_g[2]),
                                padding='same')(phi_g)  

  add_xg = Add()([upsampled_g, theta_x])
  add_xg = Activation('relu')(add_xg)
    
  psi = Conv2D(1, kernel_size=(1, 1), padding='same')(add_xg)
  psi = Activation('sigmoid')(psi)
  shape_psi = K.int_shape(psi)
    
  upsampled_coeffs = UpSampling2D((shape_x[1]//shape_psi[1], shape_x[2]//shape_psi[2]))(psi)  

  y = Multiply()([upsampled_coeffs, x])

  result = Conv2D(shape_x[3], kernel_size=(1, 1), padding='same')(y)
  result = BatchNorm()(result)
  result = Activation('relu')(result)
  return result

def GatingSignal(layer):
  """ this is simply 1x1 convolution, batchnorm, activation """
  layer_shape = K.int_shape(layer)
  x = Conv2D(layer_shape[3], kernel_size=(1, 1), strides=(1, 1), padding='same')(layer)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  return x
    

def Attn_UNet(input_shape, num_channels, num_classes, final_activation_fn):
  input_layer = Input((input_shape,input_shape,num_channels))
    
  conv1 = ConvBlock(layer=input_layer, filters=64, kernel_size=(3,3), strides=(1,1), pad_type='same')
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = ConvBlock(layer=pool1, filters=128, kernel_size=(3,3), strides=(1,1), pad_type='same')
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = ConvBlock(layer=pool2, filters=256, kernel_size=(3,3), strides=(1,1), pad_type='same')
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  conv4 = ConvBlock(layer=pool3, filters=512, kernel_size=(3,3), strides=(1,1), pad_type='same')
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
  conv5 = ConvBlock(layer=pool4, filters=1024, kernel_size=(3,3), strides=(1,1), pad_type='same')
   
  gate1 = GatingSignal(conv5)
  attn1 = AttnGatingBlock(conv4, gate1, 512)
  tr1 = Conv2DTranspose(512, kernel_size=(3,3), strides=(2,2), padding='same')(conv5)
  tr1 = Activation('relu')(tr1)
  concat1 = Concatenate(axis=-1)([tr1, attn1])
    
  gate2 = GatingSignal(concat1)
  attn2 = AttnGatingBlock(conv3, gate2, 256)
  tr2 = Conv2DTranspose(256, kernel_size=(3,3), strides=(2,2), padding='same')(concat1)
  tr2 = Activation('relu')(tr2)
  concat2 = Concatenate(axis=-1)([tr2, attn2])

  gate3 = GatingSignal(concat2)
  attn3 = AttnGatingBlock(conv2, gate3, 128)
  tr3 = Conv2DTranspose(128, kernel_size=(3,3), strides=(2,2), padding='same')(concat2)
  tr3 = Activation('relu')(tr3)
  concat3 = Concatenate(axis=-1)([tr3, attn3])
    
  gate4 = GatingSignal(concat3)
  attn4 = AttnGatingBlock(conv1, gate4, 64)
  tr4 = Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2), padding='same')(concat3)
  tr4 = Activation('relu')(tr4)
  concat4 = Concatenate(axis=-1)([tr4, attn4])
    
  conv6 = ConvBlock(concat4, filters=64, kernel_size=(3,3), strides=(1,1), pad_type='same')
  out = Conv2D(num_classes, kernel_size=(1, 1))(conv6)
  out = Activation(final_activation_fn)(out)
    
  model = Model(inputs=input_layer, outputs=out)
  return model
