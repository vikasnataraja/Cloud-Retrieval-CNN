import os
import numpy as np
import h5py
import cv2

def get_optical_thickness(data_dir, fnames, num_classes, dimensions=480, file_format='png',
                            save=False, save_labels_dir=None):

  # read only h5 files"
  #fnames = [file for file in os.listdir(data_dir) if file.endswith('.h5')]
  cot_bins=np.concatenate((np.arange(0.0, 1.0, 0.1),
                          np.arange(1.0, 10.0, 1.0),
                          np.arange(10.0, 20.0, 2.0),
                          np.arange(20.0, 50.0, 5.0),
                          np.arange(50.0, 101.0, 10.0)))

  pxvals = np.uint8([(num_classes/cot_bins.shape[0])*i for i in range(cot_bins.shape[0])])
    
  store_cots = {}
  for i in range(len(fnames)):
    f = h5py.File(os.path.join(data_dir,fnames[i]), 'r')

    cot = f['cot_inp_3d'][...][:, :, 0, 2]

    classmap = np.zeros((cot.shape[0],cot.shape[1]),dtype='float32')
    
    for k in range(cot_bins.shape[0]):
      try:
        classmap[np.bitwise_and(cot>=cot_bins[k],cot<cot_bins[k+1])] = pxvals[k] 
      except IndexError:
        classmap[cot>=cot_bins[k]] = pxvals[k]
      
    store_cots['{}'.format(fnames[i])] = classmap
    
    if save:            
      if not os.path.isdir(save_labels_dir):
        print('Output directory {} does not exist,'\
              ' creating it now ...'.format(save_labels_dir))
        os.makedirs(save_labels_dir)
      cv2.imwrite(os.path.join(save_labels_dir,fnames[i])+'_{}.{}'.format(i,file_format),classmap)
    
  return store_cots


def get_radiances(data_dir, fnames, dimensions=480):

  # read only h5 files"
  #fnames = [file for file in os.listdir(data_dir) if file.endswith('.h5')]
  store_rads = {}
  h5dict = {}  
  for i in range(len(fnames)):
    f = h5py.File(os.path.join(data_dir,fnames[i]), 'r')
    store_rads['{}'.format(fnames[i])] = np.float32(f['rad_mca_3d'][...][:, :, 0, 2])

  return store_rads

  