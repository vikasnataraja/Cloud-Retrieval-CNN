import numpy as np
from imageio import imread, imwrite
import argparse
import os

def __yes_or_no(question):
    reply = str(input(question + ' (y/n): ')).lower().strip()
    if (reply[0] == 'y') or (reply[0] == 'yes'):
        return True
    if (reply[0] == 'n') or (reply[0] == 'no'):
        return False

def crop_and_save(img, fname_count, args):

    if len(img.shape) == 2:
        imgwidth, imgheight = img.shape
    else:
        imgwidth, imgheight, _ = img.shape
    
    for i in range(0,imgwidth,int(args.crop_dims[0]/2)):
        for j in range(0, imgheight,int(args.crop_dims[1]/2)):
            if len(img.shape)==2:
                cropped_img = img[i:i+args.crop_dims[0],j:j+args.crop_dims[1]]  
            else:
                cropped_img = img[i:i+args.crop_dims[0],j:j+args.crop_dims[1],:]
            #print(cropped_img.shape[:2],args.crop_dims[:2])
            if cropped_img.shape[:2] == args.crop_dims[:2]:
                save_name = os.path.join(args.output_dir,args.fname_prefix)+'_000{}.{}'.format(fname_count,args.file_format)
                #print(save_name)
                imwrite(uri=save_name, im=cropped_img, format=args.file_format)
                fname_count += 1
    if not len(os.listdir(args.output_dir)):
        print('Something went wrong, try again\n')
        exit()
    else:
        print('Saved {} images in {}\n'.format(fname_count+1,args.output_dir))
        return fname_count
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', default='original_images', type=str, 
                        help="Path to original images directory")
    parser.add_argument('--output_dir', default='train_images', type=str, 
                        help="Path to the output directory where images will be saved")
    parser.add_argument('--crop_dims', default=(64,64), type=tuple, 
                        help="Crop dimensions tuple of width x height")
    parser.add_argument('--fname_prefix', default='image', type=str, 
                        help="Prefix for the name of the saved images. E.g: 'prefix_0.png'")
    parser.add_argument('--file_format', default='png', type=str, 
                        help="Choose between png, jpg, jpeg or any other standard image formats")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print('Output directory {} does not exist,'\
              ' creating it now ...'.format(args.output_dir))
        os.makedirs(args.output_dir)
    
    print('Images will be saved as {}.{} and will be of' \
          ' size {}'.format(os.path.join(args.output_dir,args.fname_prefix)+'_xxx',args.file_format, args.crop_dims))
    if __yes_or_no('Continue?') is True: 
        fname_count = 0
        for imagepath in os.listdir(args.image_dir):
            image = imread(os.path.join(args.image_dir,imagepath))
            fname_count = crop_and_save(image, fname_count, args)
    else:
        print('Aborting ...\n')
        exit()
