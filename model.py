import tensorflow as tf
import numpy as np
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, Dropout
from keras.layers import Lambda
from keras.layers import Concatenate, Add
from keras.models import Model

def BatchNorm():
  return BatchNormalization(momentum=0.95, epsilon=1e-5)

def common_skip(prev, num_filters, kernel_size, 
              stride_tuple, pad_type, atrous_rate, name):
  """
  The common ResNet block shared by the identity block
  and the convolutional block. Both of those blocks share
  this common functionality.
  """
  x1 = Conv2D(filters=num_filters, kernel_size=kernel_size, 
              strides=stride_tuple, dilation_rate=atrous_rate,
              padding=pad_type, use_bias=True)(prev)
  x1 = BatchNorm()(x1)
  x1 = Activation('relu')(x1)
  if name=="halve_feature_map":
      x1 = MaxPooling2D(pool_size=(2,2),padding='same')(x1)
  # dropout rate is 10%
  x1 = Dropout(rate=0.1)(x1)
  x2 = Conv2D(num_filters, kernel_size=kernel_size, 
              strides=stride_tuple, dilation_rate=atrous_rate,
              padding=pad_type, use_bias=True)(x1)
  x2 = BatchNorm()(x2)

  return x2

def convolution_branch(prev, num_filters, kernel_size, 
                     stride_tuple, pad_type, atrous_rate, name):

  prev = Conv2D(num_filters, kernel_size=kernel_size, strides=stride_tuple,
                padding=pad_type, dilation_rate=atrous_rate, use_bias=True)(prev)
  
  prev = BatchNorm()(prev)
  if name=='halve_feature_map':
      # halve the size of feature map by using same padding, 2x2 pooling
    prev = MaxPooling2D(pool_size=(2,2),padding='same')(prev)  
  return prev


def empty_branch(prev):
  return prev

def convolutional_resnet_block(prev_layer, num_filters, name, kernel_size,
                             stride_tuple, pad_type, atrous_rate=1):

  #prev_layer = BatchNorm()(prev_layer)
  prev_layer = Activation('relu')(prev_layer)
  
  block_1 = common_skip(prev=prev_layer, num_filters=num_filters, 
                        name=name, kernel_size=kernel_size, 
                        stride_tuple=stride_tuple,
                        pad_type=pad_type,
                        atrous_rate=atrous_rate)

  block_2 = convolution_branch(prev=prev_layer, num_filters=num_filters,
                               kernel_size=kernel_size, 
                               stride_tuple=stride_tuple,
                               pad_type=pad_type,
                               atrous_rate=atrous_rate,
                               name=name)
  
  added = Add()([block_1, block_2])
  
  return added
  

def identity_resnet_block(prev_layer, num_filters, name, kernel_size,
                        stride_tuple, pad_type, atrous_rate=1):
  
  #prev_layer = BatchNorm()(prev_layer)
  prev_layer = Activation('relu')(prev_layer)
  
  block_1 = common_skip(prev=prev_layer, num_filters=num_filters, 
                        kernel_size=kernel_size, 
                        stride_tuple=stride_tuple,
                        pad_type=pad_type, 
                        atrous_rate=atrous_rate,
                        name=name)
  
  block_2 = empty_branch(prev_layer)

  added = Add()([block_1, block_2])
  
  return added


def ResNet(input_layer):
  
  x = Conv2D(16, (3, 3), strides=(1, 1), padding='same',
             use_bias=True)(input_layer)
  x = BatchNorm()(x)
  #print('conv',input_layer.shape)
  x = identity_resnet_block(x, num_filters=16, kernel_size=(3,3),
                            stride_tuple=(1,1), name="identity",
                            pad_type='same', atrous_rate=1)
  
  x = convolutional_resnet_block(x, num_filters=32, kernel_size=(3,3),
                                 stride_tuple=(1,1), name="halve_feature_map",
                                 pad_type='same', atrous_rate=1)
  
  x = convolutional_resnet_block(x, num_filters=64, kernel_size=(3,3),
                                 stride_tuple=(1,1), name="halve_feature_map", 
                                 pad_type='same', atrous_rate=1)
  
  x = identity_resnet_block(x, num_filters=64, kernel_size=(3,3),
                            stride_tuple=(1,1), name="identity",
                            pad_type='same', atrous_rate=1)
  
  x = convolutional_resnet_block(x, num_filters=128, kernel_size=(3,3),
                                 stride_tuple=(1,1), name="halve_feature_map", 
                                 pad_type='same', atrous_rate=1)
  
  x = identity_resnet_block(x, num_filters=128, kernel_size=(3,3),
                            stride_tuple=(1,1), name="identity",
                            pad_type='same', atrous_rate=1)
  
  """ dilated/atrous convolutional ResNet block starts here"""
  
  x = convolutional_resnet_block(x, num_filters=256, kernel_size=(3,3),
                                 stride_tuple=(1,1), name="full_feature_map", 
                                 pad_type='same', atrous_rate=2)
  
  x = identity_resnet_block(x, num_filters=256, kernel_size=(3,3),
                            stride_tuple=(1,1), name="identity",
                            pad_type='same', atrous_rate=2)
  
  x = convolutional_resnet_block(x, num_filters=512, kernel_size=(3,3),
                                 stride_tuple=(1,1), name="full_feature_map", 
                                 pad_type='same', atrous_rate=4)
  
  x = identity_resnet_block(x, num_filters=512, kernel_size=(3,3),
                            stride_tuple=(1,1), name="identity",
                            pad_type='same', atrous_rate=4)
  
  #x = BatchNorm()(x)
  x = Activation('relu')(x)

  x = Conv2D(filters=512,kernel_size=(3,3),
             strides=(1,1), padding='same',dilation_rate=2)(x)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(filters=512,kernel_size=(3,3),
             strides=(1,1), padding='same',dilation_rate=2)(x)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  
  """End of dilated convolution block"""
  
  x = Conv2D(filters=512,kernel_size=(3,3),
             strides=(1,1), padding='same',dilation_rate=1)(x)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(filters=512,kernel_size=(3,3),
             strides=(1,1), padding='same',dilation_rate=1)(x)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  
  print('Finished building ResNet')
  return x
  
"""Spatial Pyramid Pooling"""

def upsample_bilinear(in_tensor, new_size):
  resized_height, resized_width = new_size
  return tf.image.resize(images=in_tensor,
                         size=[resized_height,resized_width],
                         method='bilinear',
                         align_corners=True)

def spp_block(prev_layer, pool_size_int, feature_map_shape):

  #kernel = [(1,1),(2,2),(4,4),(8,8)]
  #strides = [(1,1),(2,2),(4,4),(8,8)]
  pool_size_tuple = (pool_size_int, pool_size_int)
  pool_layer = AveragePooling2D(pool_size=pool_size_tuple)(prev_layer)
  conv1 = Conv2D(128, (1, 1), strides=(1, 1),
                      use_bias=True)(pool_layer)
  conv1 = BatchNorm()(conv1)
  conv1 = Activation('relu')(conv1)
  
  # upsampling
  upsampled_layer = Lambda(upsample_bilinear, 
                           arguments={'new_size':feature_map_shape})(conv1)
  
  return upsampled_layer

def pyramid_pooling_module(resnet_last, output_shape):
  """Build the Pyramid Pooling Module."""
  
  # feature map size to be used for interpolation
  # for 128x128 image, this will need to be doubled to (16,16)
  # need to automate this with a ratio
  feature_map_size = (int(output_shape/8),int(output_shape/8)) # (height, width) not (width, height)
  pool_sizes = [1,2,4,8]

  pool_block1 = spp_block(resnet_last, pool_sizes[0], feature_map_size)
  pool_block2 = spp_block(resnet_last, pool_sizes[1], feature_map_size)
  pool_block4 = spp_block(resnet_last, pool_sizes[2], feature_map_size)
  pool_block8 = spp_block(resnet_last, pool_sizes[3], feature_map_size)

  # concat all these layers with previous layer. resulted
  # shape=(None,feature_map_size_x,feature_map_size_y,4096)
  #print('shape of resnet is',resnet_last.shape)
  concat = Concatenate()([resnet_last,
                          pool_block8,
                          pool_block4,
                          pool_block2,
                          pool_block1])
  return concat    
  

"""
Deconvolution layer comes after concatenation of SPP layers
https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/layers/Conv2DTranspose
From the paper:
"Finally, multi-scale features are fused to obtain an image with 
the same size as the input image by the transposed convolution"
"""
def deconvolution_module(concat_layer, num_classes, output_shape):
  
  deconv_layer = Conv2DTranspose(filters=num_classes, kernel_size=(16,16),
                                 strides=(1,1),padding='same')(concat_layer)
  
  # output shape needs to be 128,128, so upsample from 16x16
  deconv_layer = Lambda(upsample_bilinear,
                        arguments={'new_size':output_shape})(deconv_layer)
  
  #deconv_layer.set_shape((None,128,128,1))
  deconv_layer = BatchNorm()(deconv_layer)

  deconv_layer = Activation('softmax')(deconv_layer)
                        
  return deconv_layer