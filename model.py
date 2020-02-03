import tensorflow as tf
import keras
import numpy as np
from keras import layers
from keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, Dropout
from keras.layers import ZeroPadding2D, Lambda

# might be keras.layers.merge import Add Concatenate
from keras.layers import Concatenate, Add
from keras.models import Model


def BatchNorm():
    return BatchNormalization(momentum=0.95, epsilon=1e-5)


def upsample_bilinear(in_tensor, new_size):
    resized_height, resized_width = new_size
    return tf.image.resize(images=in_tensor,size=[resized_height,resized_width],
                           method=ResizeMethodV1.BILINEAR,
                           align_corners=True)

def spp_block(prev_layer, pool_size, feature_map_shape):

    #kernel = [(1,1),(2,2),(4,4),(8,8)]
    #strides = [(1,1),(2,2),(4,4),(8,8)]
    pool_size_tuple = (pool_size, pool_size)
    prev_layer = AveragePooling2D(pool_size=pool_size_tuple)
    prev_layer = Conv2D(128, (1, 1), strides=(1, 1),
                        use_bias=False)(prev_layer)
    prev_layer = BatchNorm(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    
    # prev_layer = Lambda(Interp, arguments={
    #                    'shape': feature_map_shape})(prev_layer)
    
    # upsampling
    prev_layer = upsample_bilinear(prev_layer,feature_map_shape)
    #prev_layer = UpSampling2D(size=(2, 2),
    #                          interpolation='bilinear')

    return prev_layer

def common_skip(prev, num_filters, atrous_rate, 
                kernel_size, stride_tuple, name):
    """
    The common ResNet block shared by the identity block
    and the convolutional block. Both of those blocks share
    this common functionality.
    """

    prev = BatchNorm()(prev)
    prev = Activation('relu')(prev)
    # prev = ZeroPadding2D(padding=(pad, pad))(prev)
    
    """Syntax:
    keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), 
                    padding='valid', data_format=None, dilation_rate=(1, 1),
                    activation=None, use_bias=True)
    """
    prev = Conv2D(num_filters, kernel_size=kernel_size, 
                  strides=stride_tuple, dilation_rate=atrous_rate,
                  use_bias=False)(prev)
    if name=="halve_feature_map":
        prev = MaxPooling2D(pool_size=(2,2),padding='valid')

    prev = BatchNorm()(prev)
    prev = Activation('relu')(prev)
    # dropout rate is 10%
    prev = Dropout(rate=0.1)
    prev = Conv2D(num_filters, kernel_size=kernel_size, 
                  strides=stride_tuple, dilation_rate=atrous_rate,
                  use_bias=False)(prev)
    
    # I think this should be followed by BatchNorm and an activation
    return prev

def convolution_branch(name, num_filters, kernel_size, prev):

    prev = Conv2D(num_filters, kernel_size=(1, 1), strides=(2, 2),
                      use_bias=False)(prev)
    
    if name=='halve_feature_map':
        # halve the size of feature map by using valid padding, 2x2 pooling
        prev = MaxPooling2D(pool_size=(2,2),padding='valid')

    prev = BatchNorm()(prev)
    return prev


def empty_branch(prev):
    return prev

def convolutional_resnet_block(prev_layer, num_filters, name, kernel_size,
                               stride_tuple,atrous_rate=1):
   
    prev_layer = Activation('relu')(prev_layer)
    block_1 = common_skip(prev_layer, num_filters, name, kernel_size=kernel_size, 
                          stride_tuple=stride_tuple,
                          atrous_rate=atrous_rate)

    block_2 = convolution_branch(name, num_filters=num_filters,
                                 kernel_size=kernel_size, 
                                 prev_layer=prev_layer)
    added = Add()([block_1, block_2])
    
    return added
    

def identity_resnet_block(prev_layer, num_filters, name, kernel_size,
                          stride_tuple, atrous_rate=1):
    
    prev_layer = Activation('relu')(prev_layer)

    block_1 = common_skip(prev_layer, num_filters, name, kernel_size=kernel_size, 
                          stride_tuple=stride_tuple,
                          atrous_rate=atrous_rate)
    
    block_2 = empty_branch(prev_layer)
    
    added = Add()([block_1, block_2])
    
    return added


def ResNet(input_layer):
    
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same',
                  use_bias=False)(input_layer)
    
    x = identity_resnet_block(x, num_filters=16, kernel_size=(3,3),
                              stride_tuple=(1,1), name="identity",
                              atrous_rate=1)
    
    x = convolutional_resnet_block(x, num_filters=32, kernel_size=(3,3),
                                   stride_tuple=(1,1), name="halve_feature_map",
                                   atrous_rate=1)
    
    x = convolutional_resnet_block(x, num_filters=64, kernel_size=(3,3),
                                   stride_tuple=(1,1), name="halve_feature_map", 
                                   atrous_rate=1)
    
    x = identity_resnet_block(x, num_filters=64, kernel_size=(3,3),
                              stride_tuple=(1,1), name="identity",
                              atrous_rate=1)
    
    x = convolutional_resnet_block(x, num_filters=128, kernel_size=(3,3),
                                   stride_tuple=(1,1), name="halve_feature_map", 
                                   atrous_rate=1)
    
    x = identity_resnet_block(x, num_filters=128, kernel_size=(3,3),
                              stride_tuple=(1,1), name="identity",
                              atrous_rate=1)
    
    """ dilated/atrous convolutional ResNet block starts here"""
    
    x = convolutional_resnet_block(x, num_filters=256, kernel_size=(3,3),
                                   stride_tuple=(1,1), name="full_feature_map", 
                                   atrous_rate=2)
    
    x = identity_resnet_block(x, num_filters=256, kernel_size=(3,3),
                              stride_tuple=(1,1), name="identity",
                              atrous_rate=2)
    
    x = convolutional_resnet_block(x, num_filters=512, kernel_size=(3,3),
                                   stride_tuple=(1,1), name="full_feature_map", 
                                   atrous_rate=4)
    
    x = identity_resnet_block(x, num_filters=512, kernel_size=(3,3),
                              stride_tuple=(1,1), name="identity",
                              atrous_rate=4)
    #Syntax:
    #keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), 
    #                padding='valid', data_format=None, dilation_rate=(1, 1),
    #                activation=None, use_bias=True)
    
    x = Conv2D(filters=512,kernel_size=(3,3),
               strides=(1,1), padding='valid',dilation_rate=2)(x)
    x = BatchNorm()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=512,kernel_size=(3,3),
               strides=(1,1), padding='valid',dilation_rate=2)(x)
    x = BatchNorm()(x)
    x = Activation('relu')(x)
    
    """End of dilated convolution block"""
    
    x = Conv2D(filters=512,kernel_size=(3,3),
               strides=(1,1), padding='valid',dilation_rate=1)(x)
    x = BatchNorm()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=512,kernel_size=(3,3),
               strides=(1,1), padding='valid',dilation_rate=1)(x)
    x = BatchNorm()(x)
    x = Activation('relu')(x)
    
    return x
    
"""Spatial Pyramid Pooling"""
def build_pyramid_pooling_module(resnet_last):
    """Build the Pyramid Pooling Module."""
    
    # feature map size to be used for interpolation
    feature_map_size = (16,16) # (height, width) not (width, height)
    pool_sizes = [1,2,4,8]

    pool_block1 = spp_block(resnet_last, pool_sizes[0], feature_map_size)
    pool_block2 = spp_block(resnet_last, pool_sizes[1], feature_map_size)
    pool_block4 = spp_block(resnet_last, pool_sizes[2], feature_map_size)
    pool_block8 = spp_block(resnet_last, pool_sizes[3], feature_map_size)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
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
def add_deconvolution_layer(concat_layer):
    deconv_layer = Conv2DTranspose(filters=1,kernel_size=(16,16))
    return deconv_layer