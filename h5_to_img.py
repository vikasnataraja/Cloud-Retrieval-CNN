import os
import numpy as np
import h5py
import cv2
import pandas as pd


def store_optical_thickness(data_dir, dimensions, fname_prefix='image', file_format='png',
                            save=False, save_labels_dir=None):

  # read only h5 files"
  fnames = [file for file in os.listdir(data_dir) if file.endswith('.h5')]
  cot_bins=np.concatenate((np.arange(0.0, 1.0, 0.1),
                          np.arange(1.0, 10.0, 1.0),
                          np.arange(10.0, 20.0, 2.0),
                          np.arange(20.0, 50.0, 5.0),
                          np.arange(50.0, 101.0, 10.0)))

  pxvals = [(255/cot_bins.shape[0])*i for i in range(cot_bins.shape[0])]
  
  print('Images will be saved as {}_0xx.{} in specified dirs\n'.format(fname_prefix,file_format))
  
  store_cots = np.zeros((len(fnames),dimensions,dimensions),np.float32)
  for i in range(len(fnames)):
    f = h5py.File(os.path.join(data_dir,fnames[i]), 'r')

    cot = f['cot_inp_3d'][...][:, :, 0, 2]

    classmap = np.zeros((cot.shape[0],cot.shape[1]),dtype='float32')
    
    for k in range(cot_bins.shape[0]):
      for row in range(cot.shape[0]):
        for col in range(cot.shape[1]):
          if cot[row,col]>100:
            cot[row,col] = 100
          try:
            if (cot[row,col]>=cot_bins[k]) and (cot[row,col] <= cot_bins[k+1]):
              classmap[row,col] = pxvals[k]
          except IndexError:
            pass
    store_cots[i] = classmap
    if save:            
      if not os.path.isdir(save_labels_dir):
        print('Output directory {} does not exist,'\
              ' creating it now ...'.format(save_labels_dir))
        os.makedirs(save_labels_dir)
      cv2.imwrite(os.path.join(save_labels_dir,fnames[i])+'_{}.{}'.format(i,file_format),classmap)
    
  return store_cots


def store_radiances(data_dir, dimensions, fname_prefix='image'):

  # read only h5 files"
  fnames = [file for file in os.listdir(data_dir) if file.endswith('.h5')]
  store_rads = np.zeros((len(fnames),dimensions,dimensions),np.float32)
  h5dict = {}  
  for i in range(len(fnames)):
    f = h5py.File(os.path.join(data_dir,fnames[i]), 'r')
    store_rads[i] = f['rad_mca_3d'][...][:, :, 0, 2]
    h5dict['{}_{}_{}'.format(fnames[i],fname_prefix,i)] = [np.array(f['rad_mca_3d'][...][:, :, 0, 2],'float32')]
  df = pd.DataFrame.from_dict(h5dict).transpose()
  df = df.reset_index()
  df.columns = ['filename','data']
  df.to_csv('training.csv')
  return store_rads

  