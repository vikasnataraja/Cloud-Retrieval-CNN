import tensorflow as tf
import numpy as np
from keras.layers import MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, Dropout
from keras.layers import Lambda
from keras.layers import Concatenate, Add
from keras.models import Model
import keras

def BatchNorm():
  return BatchNormalization(momentum=0.95, epsilon=1e-5)

class UpSample(keras.layers.Layer):

  def __init__(self, new_size, **kwargs):
    self.new_size = new_size
    super(UpSample, self).__init__(**kwargs)

  def build(self, input_shape):
    super(UpSample, self).build(input_shape)

  def call(self, inputs, **kwargs):
    resized_height, resized_width = self.new_size
    return tf.image.resize(images=inputs,
                           size=[resized_height,resized_width],
                           method='bilinear',
                           align_corners=True)

  def compute_output_shape(self, input_shape):
    return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

  def get_config(self):
    config = super(UpSample, self).get_config()
    config['new_size'] = self.new_size
    return config


def common_skip(prev, num_filters, kernel_size, 
              stride_tuple, pad_type, atrous_rate, name):
  """
  The common ResNet block shared by the identity block
  and the convolutional block. Both of those blocks share
  this common functionality.
  """
  if name=='halve_feature_map':
    x1 = Conv2D(num_filters, kernel_size=kernel_size, 
                strides=(stride_tuple[0]*2,stride_tuple[1]*2), 
		dilation_rate=atrous_rate,
                padding=pad_type, use_bias=True)(prev)
  else:
    x1 = Conv2D(num_filters, kernel_size=kernel_size,
                strides=stride_tuple, dilation_rate=atrous_rate,
                padding=pad_type, use_bias=True)(prev)
  x1 = BatchNorm()(x1)
  x1 = Activation('relu')(x1)
  
  # dropout rate is 10%
  x1 = Dropout(rate=0.1)(x1)
  x2 = Conv2D(num_filters, kernel_size=kernel_size, 
              strides=stride_tuple, dilation_rate=atrous_rate,
              padding=pad_type, use_bias=False)(x1)
  x2 = BatchNorm()(x2)
  return x2

def convolution_branch(prev, num_filters, kernel_size, 
                     stride_tuple, pad_type, atrous_rate, name):
  
  if name=='halve_feature_map':
    prev = Conv2D(num_filters, kernel_size=kernel_size, strides=(stride_tuple[0]*2,stride_tuple[1]*2),
                  padding=pad_type, dilation_rate=atrous_rate, use_bias=False)(prev)
  else:
    prev = Conv2D(num_filters, kernel_size=kernel_size, strides=stride_tuple,
                  padding=pad_type, dilation_rate=atrous_rate, use_bias=False)(prev)
  prev = BatchNorm()(prev)
  return prev

def empty_branch(prev):
  return prev

def convolutional_resnet_block(prev_layer, num_filters, name, kernel_size,
                             stride_tuple, pad_type, atrous_rate):

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
                        stride_tuple, pad_type, atrous_rate):
  
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
  
  x = Conv2D(16, (7, 7), strides=(1, 1), padding='same',
             use_bias=False)(input_layer)
  x = BatchNorm()(x)
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
  
  x = Activation('relu')(x)

  x = Conv2D(filters=512,kernel_size=(3,3),
             strides=(1,1), padding='valid',dilation_rate=2,use_bias=False)(x)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  x = ZeroPadding2D(padding=(2,2))(x)
    
  x = Conv2D(filters=512,kernel_size=(3,3),
             strides=(1,1), padding='valid',dilation_rate=2, use_bias=False)(x)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  x = ZeroPadding2D(padding=(2,2))(x)
  
  """End of dilated convolution block"""
  
  x = Conv2D(filters=512,kernel_size=(3,3),
             strides=(1,1), padding='same',dilation_rate=1,use_bias=False)(x)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(filters=512,kernel_size=(3,3),
             strides=(1,1), padding='same',dilation_rate=1, use_bias=False)(x)
  x = BatchNorm()(x)
  x = Activation('relu')(x)
  return x
  
"""Spatial Pyramid Pooling"""

def upsample_bilinear(in_tensor, new_size):
  resized_height, resized_width = new_size
  return tf.image.resize(images=in_tensor,
                         size=[resized_height,resized_width],
                         method='bilinear',
                         align_corners=True)

def spp_block(prev_layer, pool_size_int, feature_map_shape):
  pool_size_tuple = (pool_size_int, pool_size_int)
  pool_layer = AveragePooling2D(pool_size=pool_size_tuple, strides=pool_size_tuple)(prev_layer)
  conv1 = Conv2D(128, (1, 1), strides=(1, 1),
                 use_bias=False)(pool_layer)
  conv1 = BatchNorm()(conv1)
  conv1 = Activation('relu')(conv1)
  
  # upsampling
  upsampled_layer = UpSample(new_size=feature_map_shape)(conv1)
  # upsampled_layer = Lambda(upsample_bilinear, 
  #                          arguments={'new_size':feature_map_shape})(conv1)
  return upsampled_layer

def pyramid_pooling_module(resnet_last, output_shape, pool_sizes):
  """Build the Pyramid Pooling Module."""
  
  # feature map size to be used for interpolation
  feature_map_size = (int(output_shape/8),int(output_shape/8)) # (height, width) not (width, height)

  pool_block1 = spp_block(resnet_last, pool_sizes[0], feature_map_size)
  pool_block2 = spp_block(resnet_last, pool_sizes[1], feature_map_size)
  pool_block3 = spp_block(resnet_last, pool_sizes[2], feature_map_size)
  pool_block4 = spp_block(resnet_last, pool_sizes[3], feature_map_size)

  # concatenate all these layers with previous layer. resulted
  # shape=(batch_size,feature_map_size_x,feature_map_size_y,4096)
  concat = Concatenate(axis=-1)([resnet_last,
                          	 pool_block4,
                                 pool_block3,
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
def deconvolution_module(concat_layer, num_classes, out_shape, activation_fn, transpose=True):
  if transpose:
    x = Conv2DTranspose(filters=num_classes, kernel_size=(int(out_shape[0]/8),int(out_shape[1]/8)),
               	        strides=(1,1), use_bias=False)(concat_layer)
  else:
    x = Conv2D(filters=num_classes, kernel_size=(1,1),
               strides=(1,1), padding='same', use_bias=False)(concat_layer)
  # upsample to output_shape
  x = UpSample(new_size=out_shape)(x)
  # x = Lambda(upsample_bilinear,
  #            arguments={'new_size':out_shape})(x)
  x = Activation(activation_fn)(x)
  return x
