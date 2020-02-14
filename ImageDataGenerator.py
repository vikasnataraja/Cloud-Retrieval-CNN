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
import numpy as np

class ImageGenerator(Sequence):
    
    def __init__(self, image_list=None, label_list=None, 
                 image_dir='./train_images', anno_dir='./labels',num_classes = 50, 
                 batch_size = 16, resize_shape_tuple=(128,128), num_channels=3,
                 separator='.',shuffle=True):
        
        self.image_list = image_list
        self.label_list = label_list
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.resize_shape_tuple = resize_shape_tuple
        self.separator = separator
        self.num_channels = num_channels
        self.shuffle = shuffle
        self.on_epoch_end()


    def __data_generator(self, batch_IDs):
        
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError('ERROR! The folder {} does not'
                                    ' exist\n'.format(self.image_dir))
        
        X = np.zeros([self.batch_size, self.resize_shape_tuple[0], 
                      self.resize_shape_tuple[1], self.num_channels])
        y = np.zeros([self.batch_size, self.resize_shape_tuple[0], 
                      self.resize_shape_tuple[1], self.num_classes])
        
        for i, vals in enumerate(batch_IDs):
            img = resize(imread(os.path.join(self.image_dir, self.image_list[i])), 
                         self.resize_shape_tuple)
            label = resize(imread(os.path.join(self.anno_dir, self.label_list[i])),
                       self.resize_shape_tuple)
            #print(y.shape,n_classes)
            label = (np.arange(self.num_classes) == label[:,:,None]).astype('float32')
            #print(y.shape,n_classes)
            assert label.shape[2] == self.num_classes,"Error, dimensions do not match"
            
            X[i,] = img
            
            y[i] = label
            
            #print(X.shape,y.shape)
            return X, y
                    
    
    def __len__(self):
            
        return int(np.floor(len(self.image_list)/self.batch_size))

    def __getitem__(self, index):
        
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        batch_IDs = [self.image_list[k] for k in indices]
        # Generate data
        X, y = self.__data_generator(batch_IDs)

        return X, y
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_list))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    