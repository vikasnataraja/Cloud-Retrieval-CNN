import numpy as np
import tensorflow as tf
from utils.losses import focal_loss
from keras.utils import Sequence
from keras.utils import to_categorical
from keras.models import load_model
import os
import h5py
import cv2


def normalize_img(img):
  return (img-img.min())/(img.max()-img.min())

def standard_normalize(img):
  if img.std()!=0.:
    img = img.astype('float32')
    means = img.mean(axis=(0,1), dtype='float64')
    devs = img.std(axis=(0,1), dtype='float64')
    img = (img - means)/devs
    img = np.clip(img, -1.0, 1.0)
    img = (img + 1.0)/2.0
  return img

def resize_img(img, resize_dims):
  return cv2.resize(img, resize_dims)
 
def preprocess(img, resize_dims=None, normalize=False):
  """ Pre-processing steps for an image """
  if normalize:
    img = standard_normalize(img)
  if resize_dims is not None:
    img = resize_img(img, resize_dims)

  if len(img.shape) > 2:
    img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
  else:
    img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
  return img
  

class ImageGenerator(Sequence):
  """ 
  Adapted from sources: 
  https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow/blob/master/utils/preprocessing.py
  https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
  """
  def __init__(self, image_list, image_dict, label_dict,
               num_classes, batch_size, input_shape, output_shape,
               num_channels, augment, normalize, 
  	       to_fit, shuffle, augmentation):
      
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
    if self.to_fit:
      self.on_epoch_end()

  def _data_generator(self, batch_images):
    X = np.zeros((self.batch_size, self.input_shape,
                  self.input_shape, self.num_channels))
    y = np.zeros((self.batch_size, self.output_shape,
                  self.output_shape, self.num_classes)) 

    for i, val in enumerate(batch_images):
      # read in input image from image dictionary
      img = self.image_dict[val]
      # read in ground truth (mask) from label dictionary
      label = self.label_dict[val]
      if self.normalize:
        img = standard_normalize(img)
      # if self.augment:
      #   augmented = self.augmentation(image=img,mask=label)
      #   img = augmented['image']
      #   label = augmented['mask']
      img = np.reshape(img, (img.shape[0], img.shape[1], self.num_channels))
      X[i] = img
      
      # one-hot encoding of mask labels using Keras. This will transform mask from 
      # (width x height) to (width x height x num_classes) with 1s and 0s
      label = np.uint8(to_categorical(label, num_classes=self.num_classes))
      y[i] = label
    return X,y

  def __len__(self):
    return int(np.floor(len(self.image_list)/self.batch_size))
  
  def __getitem__(self, index):
    indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
    batch_images = [self.image_list[k] for k in indices]
 
    # Generate data and ground truth
    X,y = self._data_generator(batch_images)
    return X,y
  
  def on_epoch_end(self):
    self.indices = np.arange(len(self.image_list))
    if self.shuffle:
      np.random.shuffle(self.indices)


cot_bins = np.concatenate((np.arange(0.0, 1.0, 0.1),
                           np.arange(1.0, 10.0, 1.0),
                           np.arange(10.0, 20.0, 2.0),
                           np.arange(20.0, 50.0, 5.0),
                           np.arange(50.0, 101.0, 10.0)))

pxvals = np.arange(0, cot_bins.shape[0]) 


def get_data(fdir, rad_keyname, cot_keyname):
  """ Combine h5 files to get radiance and COT data as dictionaries 
  Args:
    - fdir: str, directory containing h5 files.
    - rad_keyname: str, key for radiance data in h5 files.
    - cot_keyname: str, key for COT data in h5 files.

  Returns:
    - store_rads: dict, dictionary containing radiance data.
    - store_cots: dict, dictionary containing COT data.
  
  """

  fnames = [file for file in os.listdir(fdir) if file.endswith('.h5')]
  store_rads = {}
  store_cots = {}
  
  for i in range(len(fnames)):
    f = h5py.File(os.path.join(fdir,fnames[i]), 'r')
    
    if len(f) <= 3: # coarsened data files only have 3 keys and are stored in different format
      cot = f['{}'.format(cot_keyname)][...][:, :, 0, 2]
      classmap = np.zeros((cot.shape[0],cot.shape[1]),dtype='uint8')
      for k in range(pxvals.size):
        if k < (pxvals.size-1):
          classmap[np.bitwise_and(cot>=cot_bins[k], cot<cot_bins[k+1])] = pxvals[k] 
        else:
          classmap[cot>=cot_bins[k]] = pxvals[k]

      store_cots['{}'.format(fnames[i])] = classmap
      rad = f['{}'.format(rad_keyname)][...]
      store_rads['{}'.format(fnames[i])] = np.float32(rad[:, :, 0, 2])
    
    else: # for original data/7 SEAS data
      cot = f['{}'.format(cot_keyname)]
      classmap = np.zeros((cot.shape[0],cot.shape[1]),dtype='uint8')
      for k in range(pxvals.size):
        if k < (pxvals.size-1):
          classmap[np.bitwise_and(cot>=cot_bins[k], cot<cot_bins[k+1])] = pxvals[k] 
        else:
          classmap[cot>=cot_bins[k]] = pxvals[k]

      store_cots['{}'.format(fnames[i])] = classmap
      rad = f['{}'.format(rad_keyname)]
      store_rads['{}'.format(fnames[i])] = np.float32(rad)

  return store_rads, store_cots


def crop_images(img_dict, crop_dims, fname_prefix):
  """ Crop images in a dictionary to given dimensions. Currently uses 50% overlap and discards right and bottom pixels.
  
  Args:
    - img_dict: dict, dictionary containing keys-value pairs of original dimensions
    - crop_dims: int, dimensions to which images must be cropped
    - fname_prefix: str, string value to use as keynames (prefix for number)
  
  Returns:
    - return_imgs: dict, dictionary containing cropped images 
  
  """
  imgs = np.array(list(img_dict.values()))
  img_names_length = len(img_dict)
  width, height = imgs.shape[1:]
  return_imgs = {}
  counter = 0
  for idx in range(img_names_length):
    for _, i in enumerate(range(0, width, int(crop_dims/2))):
      for _, j in enumerate(range(0, height, int(crop_dims/2))):
        if len(imgs.shape)>3:
          cropped_img = imgs[idx, i:i+crop_dims, j:j+crop_dims,:]
        else:
          cropped_img = imgs[idx, i:i+crop_dims, j:j+crop_dims]
        if cropped_img.shape[:2] == (crop_dims,crop_dims):
          return_imgs['{}_{}'.format(fname_prefix,counter)] = cropped_img
          counter += 1

  print('Total number of images = {}'.format(len(return_imgs)))
  return return_imgs


def get_class_space_data(input_file, file_3d, file_1d):
  radiances = np.load('{}'.format(input_file), allow_pickle=True).item()
  cot_3d = np.load('{}'.format(file_3d), allow_pickle=True).item()
  cot_1d = np.load('{}'.format(file_1d), allow_pickle=True).item()
  return radiances, cot_3d, cot_1d


def get_cot_space_data(fdir, fnames, rad_key='rad_3d', cot_true_key='cot_true', cot_1d_key='cot_1d'):
  store_rads, store_cot_true, store_cot_1d = {}, {}, {}
    
  for i in range(len(fnames)):
    f = h5py.File(os.path.join(fdir, fnames[i]), 'r')
    if len(f.keys()) > 3: # original h5 files have different keys
      rad_key = 'rad_mca_3d' # radiance key
      cot_true_key = 'cot_inp_3d' # 3D COT ground truth
      cot_1d_key = 'cot_ret_3d' # 1D COT
      store_rads['{}'.format(i)] = np.array(f[rad_key][...][:, :, 0, 2], dtype='float64')
      store_cot_true['{}'.format(i)] = np.array(f[cot_true_key][...][:, :, 0, 2], dtype='float64')
      store_cot_1d['{}'.format(i)] = np.array(f[cot_1d_key][...][:, :, 0, 2], dtype='float64')
    else:
      store_rads['{}'.format(i)] = np.array(f[rad_key], dtype='float64')
      store_cot_true['{}'.format(i)] = np.array(f[cot_true_key], dtype='float64')
      store_cot_1d['{}'.format(i)] = np.array(f[cot_1d_key], dtype='float64')

  return (crop_images(store_rads, 64, 'data'),
         crop_images(store_cot_true, 64, 'data'),
         crop_images(store_cot_1d, 64, 'data'))


def get_model(path):
  return load_model('{}'.format(path), custom_objects={"tf":tf, "focal_loss":focal_loss})



