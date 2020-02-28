import os
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import argparse

def __yes_or_no(question):
    reply = str(input(question + ' (y/n): ')).lower().strip()
    if (reply[0] == 'y') or (reply[0] == 'yes'):
        return True
    if (reply[0] == 'n') or (reply[0] == 'no'):
        return False


def convert_h5_img(args):

  # read only h5 files"
  fnames = [file for file in os.listdir(args.data_dir) if file.endswith('.h5')]
  cot_bins=np.concatenate((np.arange(0.0, 1.0, 0.1),
                          np.arange(1.0, 10.0, 1.0),
                          np.arange(10.0, 20.0, 2.0),
                          np.arange(20.0, 50.0, 5.0),
                          np.arange(50.0, 101.0, 10.0)))

  pxvals = [(255/cot_bins.shape[0])*i for i in range(cot_bins.shape[0])]
  
  print('Images will be saved as {}_0xx.{} in specified dirs\n'.format(args.fname_prefix,args.file_format))
  
  if __yes_or_no('Continue?') is True:

    for i in range(len(fnames)):
        f = h5py.File(os.path.join(args.data_dir,fnames[i]), 'r')

        cot = f['cot_inp_3d'][...][:, :, 0, 2]
        rad = f['rad_mca_3d'][...][:, :, 0, 2]

        classmap = np.zeros((cot.shape[0],cot.shape[1]),dtype='float32')
        
        for k in range(cot_bins.shape[0]):
          for row in range(cot.shape[0]):
            for col in range(cot.shape[1]):
              try:
                  if (cot[row,col]>=cot_bins[k]) and (cot[row,col] < cot_bins[k+1]):
                      classmap[row,col] = pxvals[k]
              except IndexError:
                  pass
        
        plt.imsave(os.path.join(args.save_images_dir, \
                    args.fname_prefix)+'_0{}.{}'.format(i,args.file_format),rad,cmap='gray')

        cv2.imwrite(os.path.join(args.save_labels_dir, \
                    args.fname_prefix)+'_0{}.{}'.format(i,args.file_format),classmap)
    
    if len(os.listdir(args.save_labels_dir)) == len(os.listdir(args.save_images_dir)):
      print('Saved {} images each in specified directories'.format(i))
    else:
      print('Ruh roh, something went wrong\n')
      exit()
  else:
    print('Aborting operation ...\n')
    exit(0)

if __name__=='__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('--data_dir', default='data', type=str, 
            help="Path to original hdf5 files directory")
  parser.add_argument('--save_images_dir', default='original_images', type=str, 
            help="Path to the directory where input images will be saved")
  parser.add_argument('--save_labels_dir', default='original_labels', type=str, 
            help="Path to the directory where ground truth images will be saved")
  parser.add_argument('--fname_prefix', default='image', type=str, 
            help="Prefix for the name of the saved images. E.g: 'prefix_0.png'")
  parser.add_argument('--file_format', default='png', type=str, 
            help="Choose between png, jpg, jpeg or any other standard image formats")
  args = parser.parse_args()

  if not os.path.isdir(args.save_images_dir):
        print('Output directory {} does not exist,'\
              ' creating it now ...'.format(args.save_images_dir))
        os.makedirs(args.save_images_dir)

  if not os.path.isdir(args.save_labels_dir):
        print('Output directory {} does not exist,'\
              ' creating it now ...'.format(args.save_labels_dir))
        os.makedirs(args.save_labels_dir)

  convert_h5_img(args)