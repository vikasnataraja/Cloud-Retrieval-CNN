import numpy as np
import os
import h5py
import cv2

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