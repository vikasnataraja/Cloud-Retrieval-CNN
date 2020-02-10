""" 
Modified from source: 
https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow/blob/master/utils/preprocessing.py
"""
from collections import defaultdict
import random
from skimage.io import imread
from skimage.transform import resize
from keras.utils import Sequence
import os
import numpy as np

class ImageGenerator(Sequence):
    
    def __init__(self, image_dir='./train_images', anno_dir='./labels',n_classes = 50, 
                 batch_size = 16, resize_shape_tuple=(128,128,1), 
                 separator='.', n_test=50, shuffle=True, mode='train'):
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.resize_shape_tuple = resize_shape_tuple
        self.separator = separator
        self.n_test = n_test
        self.list_IDs = os.listdir(image_dir)
        self.shuffle = shuffle
        self.mode = mode
        self.on_epoch_end()

    
    def update_inputs(self, batch_size, resize_tuple, num_classes):
        return np.zeros([batch_size, resize_tuple[0], resize_tuple[1], resize_tuple[2]]), \
               np.zeros([batch_size, resize_tuple[0], resize_tuple[1], num_classes])


    def __data_generator(self,mode):

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError('ERROR! The folder {} does not'
                                    ' exist\n'.format(self.image_dir))

        data = defaultdict(dict)

        for image_path in os.listdir(self.image_dir):
            # image number, might want to replace this or rename the images
            nmb = image_path.split(self.separator)[0]
            data[nmb]['image'] = image_path

        for anno_path in os.listdir(self.anno_dir):
            # image number, might want to replace this or rename the images
            nmb = anno_path.split(self.separator)[0]
            data[nmb]['annotation'] = anno_path
        

        values = list(data.values())

        """
        train = self.__generate(values[self.n_test:], self.n_classes, 
                               self.batch_size, self.resize_shape_tuple, 
                               self.image_dir, self.anno_dir)
        val = self.__generate(values[:self.n_test], self.n_classes, 
                               self.batch_size, self.resize_shape_tuple, 
                               self.image_dir, self.anno_dir)
        return train, val
        """
        if mode == 'train':
            return self.__generate(values[self.n_test:], self.n_classes, 
                                   self.batch_size, self.resize_shape_tuple, 
                                   self.image_dir, self.anno_dir)
        elif mode == 'valid':
            return self.__generate(values[:self.n_test], self.n_classes, 
                                   self.batch_size, self.resize_shape_tuple, 
                                   self.image_dir, self.anno_dir)
        else:
            print('ERROR! Set mode to either "train" or "valid"')
            return 0
        
    def __generate(self, values, n_classes, batch_size,
                   resize_shape_tuple, image_dir, anno_dir):
        #print('hey')
        while 1:
            #random.shuffle(values)
            images, labels = self.update_inputs(batch_size=batch_size,
                                                resize_tuple=resize_shape_tuple, 
                                                num_classes=n_classes)
            #print(labels)
            for i, vals in enumerate(values):
                img = resize(imread(os.path.join(image_dir, vals['image'])), 
                             resize_shape_tuple)
                y = resize(imread(os.path.join(anno_dir, vals['annotation'])),
                           resize_shape_tuple)
                y = (np.arange(n_classes) == y[:,:,None]).astype('float32')
                assert y.shape[2] == n_classes,"Error, dimensions do not match"
                images[i%batch_size] = img
                labels[i%batch_size] = y
                if (i+1)%batch_size == 0:
                    yield images, labels
                    images, labels = self.update_inputs(batch_size=batch_size,
                                                        resize_tuple=resize_shape_tuple, 
                                                        num_classes=n_classes)
    
    def __len__(self):
            
        return int(np.floor(len(self.list_IDs)/self.batch_size))

    def __getitem__(self, index):
        
        # Generate indexes of the batch
        #indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #batch_list_IDs = [self.list_IDs[k] for k in indices]

        # Generate data
        X, y = self.__data_generator(self.mode)

        return X, y
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    