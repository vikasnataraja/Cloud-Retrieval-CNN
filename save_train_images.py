import numpy as np
import os
import argparse
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--original_train_dir', default=None, type=str, 
                    help="Path to original training images directory")
parser.add_argument('--label_dir', default=None, type=str, 
                    help="Path to label images directory")
parser.add_argument('--output_dir', default=None, type=str, 
                    help="Path to output images directory where images will be saved")
parser.add_argument('--segmentation_format', default='jpg', type=str, 
                    help="Extension of the images to be saved")
args = parser.parse_args()

def get_common_images():
	imgs = os.listdir(args.original_train_dir)
	labels = os.listdir(args.label_dir)
	common = [os.path.splitext(label)[0]+'.'+ args.segmentation_format for label in labels]
	#print('before',common)
	#common = list(set(common) & set(imgs))
	#print('after',common)
	return common

def save_images(filename):
	img = Image.open(os.path.join(args.original_train_dir,filename))
	img.save(os.path.join(args.output_dir,filename))

def main():
	com_imgs = get_common_images()
	#print(com_imgs)
	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)
	for img in com_imgs:
		save_images(img)
	print("Finished saving images in {}".format(args.output_dir))

if __name__=='__main__':
	main()

