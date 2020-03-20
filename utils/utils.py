import numpy as np
import tensorflow as tf
from keras.utils import Sequence
from keras.utils import to_categorical
import os
import cv2

def normalize(img):
        return (img-img.min())/(img.max()-img.min())


class ImageGenerator(Sequence):
  """ 
  Adapted from sources: 
  https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow/blob/master/utils/preprocessing.py
  https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
  """
  
  def __init__(self, image_list, image_dict, label_dict,
               num_classes, batch_size, input_shape, output_shape,
               num_channels, to_fit=True, shuffle=False):
      
    self.image_list = image_list
    self.image_dict = image_dict
    self.label_dict = label_dict
    self.num_classes = num_classes
    self.batch_size = batch_size
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.num_channels = num_channels
    self.to_fit = to_fit
    self.shuffle = shuffle
    self.on_epoch_end()


  def _data_generator_X(self, batch_images):
    X = np.zeros((self.batch_size, self.input_shape,
                    self.input_shape, self.num_channels))
    
    for i, val in enumerate(batch_images):
      in_img = self.image_dict[val]
      in_img = (in_img-in_img.min())/(in_img.max()-in_img.min())
      #if self.num_channels<3:
      in_img = np.reshape(in_img, (in_img.shape[0],in_img.shape[1],self.num_channels))
      X[i] = in_img
    return X

  def _data_generator_y(self, batch_images):
    y = np.zeros((self.batch_size, self.output_shape, 
                  self.output_shape, self.num_classes))
    
    for i, val in enumerate(batch_images):
      label = self.label_dict[val]
      #print('label val\n',val)
      # one-hot encoding of mask labels using Keras. This will transform mask from 
      # (width x height) to (width x height x num_classes) with 1s and 0s
      label = np.uint8(to_categorical(label, num_classes=self.num_classes))
      y[i] = label
      
    return y
              
  
  def __len__(self):
    return int(np.floor(len(self.image_list)/self.batch_size))

  
  def __getitem__(self, index):
    indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
    
    batch_images = [self.image_list[k] for k in indices]
 
    # Generate data
    X = self._data_generator_X(batch_images)

    if self.to_fit:
      y = self._data_generator_y(batch_images)
      return X,y
    else:
      return X
  
  def on_epoch_end(self):
    self.indices = np.arange(len(self.image_list))
    if self.shuffle:
      np.random.shuffle(self.indices)