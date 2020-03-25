import numpy as np
from keras.utils import Sequence
from keras.utils import to_categorical
import os
import h5py
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
               num_channels, augment, normalize, to_fit=True, shuffle=False, augmentation=None):
      
    self.image_list = image_list
    self.image_dict = image_dict
    self.label_dict = label_dict
    self.num_classes = num_classes
    self.batch_size = batch_size
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.num_channels = num_channels
    self.to_fit = to_fit
    self.augment = augment
    self.augmentation = augmentation
    self.shuffle = shuffle
    self.normalize = normalize
    self.on_epoch_end()


  def _data_generator_X(self, batch_images):
    X = np.zeros((self.batch_size, self.input_shape,
                    self.input_shape, self.num_channels))
    
    for i, val in enumerate(batch_images):
      in_img = self.image_dict[val]
      if self.normalize:
        in_img = self.normalize_img(in_img)
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

  def normalize_img(self, img):
    return (img-img.min())/(img.max()-img.min())

def get_optical_thickness(data_dir, fnames, num_classes, file_format='png',
                            save=False, save_labels_dir=None):

  # read only h5 files"
  #fnames = [file for file in os.listdir(data_dir) if file.endswith('.h5')]
  cot_bins=np.concatenate((np.arange(0.0, 1.0, 0.1),
                          np.arange(1.0, 10.0, 1.0),
                          np.arange(10.0, 20.0, 2.0),
                          np.arange(20.0, 50.0, 5.0),
                          np.arange(50.0, 101.0, 10.0)))

  pxvals = np.arange(0,num_classes)
    
  store_cots = {}
  for i in range(len(fnames)):
    f = h5py.File(os.path.join(data_dir,fnames[i]), 'r')
    cot = f['cot_inp_3d'][...][:, :, 0, 2]
    classmap = np.zeros((cot.shape[0],cot.shape[1]),dtype='float32')
    for k in range(pxvals.size):
      if k < (pxvals.size-1):
        classmap[np.bitwise_and(cot>=cot_bins[k], cot<cot_bins[k+1])] = pxvals[k] 
      else:
        classmap[cot>=cot_bins[k]] = pxvals[k]
    store_cots['{}'.format(fnames[i])] = classmap
    
    if save:            
      if not os.path.isdir(save_labels_dir):
        print('Output directory {} does not exist,'\
              ' creating it now ...'.format(save_labels_dir))
        os.makedirs(save_labels_dir)
      cv2.imwrite(os.path.join(save_labels_dir,fnames[i])+'_{}.{}'.format(i,file_format),classmap)
    
  return store_cots


def get_radiances(data_dir, fnames):

  # read only h5 files"
  #fnames = [file for file in os.listdir(data_dir) if file.endswith('.h5')]
  store_rads = {}
  for i in range(len(fnames)):
    f = h5py.File(os.path.join(data_dir,fnames[i]), 'r')
    store_rads['{}'.format(fnames[i])] = np.float32(f['rad_mca_3d'][...][:, :, 0, 2])

  return store_rads


def crop_images(img_dict, crop_dims, fname_prefix):
  imgs = np.array(list(img_dict.values()))
  img_names = np.array(list(img_dict.keys()))
  imgwidth, imgheight = imgs.shape[1:]

  return_imgs = {}
  counter=0
  for idx in range(img_names.shape[0]):
    for hcount, i in enumerate(range(0,imgwidth,int(crop_dims/2))):
      for vcount, j in enumerate(range(0, imgheight,int(crop_dims/2))):
        cropped_img = imgs[idx, i:i+crop_dims,j:j+crop_dims]
        #print(cropped_img.shape)
        if cropped_img.shape[:2] == (crop_dims,crop_dims):
          return_imgs['{}_{}'.format(fname_prefix,counter)] = cropped_img
          counter+=1

  print('Total number of images = {}'.format(len(return_imgs)))
  return return_imgs
