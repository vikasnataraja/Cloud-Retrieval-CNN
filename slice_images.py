import numpy as np
import os

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