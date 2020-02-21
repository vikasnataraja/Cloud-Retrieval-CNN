import os
import numpy as np
import matplotlib.pyplot as plt

def convert_h5_img(args):

	# read only h5 files"
	fnames = [file for file in os.listdir(args.data_dir) if file.endswith('.h5')]
	for i in range(len(fnames)):
	    f = h5py.File(os.path.join(args.data_dir,fnames[i]), 'r')

	    cot = f['cot_inp_3d'][...][:, :, 0, 2]
	    rad = f['rad_mca_3d'][...][:, :, 0, 2]

	    # normalize
	    rad_norm = (rad - rad.min())/(rad.max()-rad.min())
	    # stack to 3d grayscale
	    rad_norm = np.dstack([rad_norm,rad_norm,rad_norm])
	    
	    plt.imsave(os.path.join(args.save_images_dir,\
	    			args.fname_prefix)+'_0{}.{}'.format(i,args.file_format),rad_norm)
	    plt.imsave(os.path.join(args.save_labels_dir,\
	    			args.fname_prefix)+'_0{}.{}'.format(i,args.file_format),cot)
	if len(os.listdir(args.save_labels_dir)) == len(os.listdir(args.save_images_dir)):
    	print('Saved {} images each in specified directories'.format(i))
	else:
		print('Uhoh, something went wrong\n')
		exit()

if __name__=='__main__':

	parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='data', type=str, 
                        help="Path to original hdf5 files directory")
    parser.add_argument('--save_images_dir', default='images', type=str, 
                        help="Path to the directory where input images will be saved")
    parser.add_argument('--save_labels_dir', default='labels', type=str, 
                        help="Path to the directory where ground truth images will be saved")
    parser.add_argument('--fname_prefix', default='image', type=str, 
                        help="Prefix for the name of the saved images. E.g: 'prefix_0.png'")
    parser.add_argument('--file_format', default='png', type=str, 
                        help="Choose between png, jpg, jpeg or any other standard image formats")
    args = parser.parse_args()

    convert_h5_img(args)