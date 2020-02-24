""" 
Adapted from sources: 
https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow/blob/master/utils/preprocessing.py
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
import random
from skimage.io import imread
from skimage.transform import resize
from keras.utils import Sequence
import os
import cv2
import numpy as np

class ImageGenerator(Sequence):
    
    def __init__(self, image_list, label_list, 
                 image_dir, anno_dir, num_classes, 
                 batch_size, resize_shape_tuple, num_channels,
                 to_fit=True, shuffle=True):
        
        self.image_list = image_list
        self.label_list = label_list
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.resize_shape_tuple = resize_shape_tuple
        self.num_channels = num_channels
        self.to_fit = to_fit
        self.shuffle = shuffle
        self.on_epoch_end()


    def _data_generator_X(self, batch_images):
        
        X = np.zeros([self.batch_size, self.resize_shape_tuple[0], 
                      self.resize_shape_tuple[1], self.num_channels])
        for i, val in enumerate(batch_images):
            
            img = cv2.resize(cv2.imread(os.path.join(self.image_dir, val)), self.resize_shape_tuple)

            X[i,] = img
            
            return X

    def _data_generator_y(self, batch_images):

        y = np.zeros([self.batch_size, self.resize_shape_tuple[0], 
                      self.resize_shape_tuple[1], self.num_classes])
        
        for i, val in enumerate(batch_images):

            # VOC labels are in png format
            label = cv2.resize(cv2.imread(os.path.join(self.anno_dir, val)),self.resize_shape_tuple)
            label = (np.arange(self.num_classes) == label[:,:,None]).astype('float32')
            assert label.shape[2] == self.num_classes,"Error, dimensions do not match"
            
            y[i] = label
            
            return y
                    
    
    def __len__(self):
            
        return int(np.floor(len(self.image_list)/self.batch_size))

    def __getitem__(self, index):
        
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        batch_images = [self.image_list[k] for k in indices]
        # Generate data
        X = self._data_generator_X(batch_images)

        if self.to_fit is True:
            y = self._data_generator_y(batch_images)
            return X,y
        else:
            return y
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_list))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    