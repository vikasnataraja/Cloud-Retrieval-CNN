import numpy as np
import os

def crop_and_save(imgs, crop_dims):
  imgwidth, imgheight = imgs.shape[1:]

  return_imgs = []
  
  for idx in range(imgs.shape[0]):
    for i in range(0,imgwidth,int(crop_dims/2)):
      for j in range(0, imgheight,int(crop_dims/2)):
        cropped_img = imgs[idx, i:i+crop_dims,j:j+crop_dims]
        #print(cropped_img.shape)
        if cropped_img.shape[:2] == (crop_dims,crop_dims):
          #save_name = os.path.join(output_dir,fname_prefix)+'_{}.{}'.format(fname_count,file_format)
          return_imgs.append(cropped_img)
          #fname_count += 1
  print('Total number of images = {}'.format(len(return_imgs)))
  return np.array(return_imgs)