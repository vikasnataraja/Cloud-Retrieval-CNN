import os
from utils.utils import get_training_data, get_rgb_radiance_data, extract_sub_patches
import argparse
import numpy as np

if __name__ =='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--fdir', default='data/', type=str, 
						help="Path to directory containing HDF5 files")
	parser.add_argument('--dest', default='data/npy', type=str, 
						help="Path to directory where npy files will be saved")
	args = parser.parse_args()
	
	if not os.path.isdir(args.fdir):
		raise OSError('\nDirectory {} does not exist, try again\n'.format(args.fdir))
	
	if not os.path.isdir(args.dest):
		print('\nDestination directory {} does not exist, creating it now...'.format(args.dest))
		os.makedirs(args.dest)

	# get training data from fdir, each dictionary will have 480x480 scenes	
	# radiance, cot_true, cot_1d = get_training_data(args.fdir, rad_keyname='rad_mca_3d', cot_true_keyname='cot_inp_3d', cot_1d_keyname='cot_ret_3d')
	radiance, cot_true, cot_1d = get_rgb_radiance_data(args.fdir)
	
	# extract 64x64 sub-patches and store in dictionary with key:value :: data_xxx: ndarray
	rad_64 = extract_sub_patches(radiance, 64, 'data', excl_borders=16) # radiance
	cot_true_64 = extract_sub_patches(cot_true, 64, 'data', excl_borders=16) # true COT
	cot_1d_64 = extract_sub_patches(cot_1d, 64, 'data', excl_borders=16) # IPA COT
	
	# save to file
	np.save(os.path.join(args.dest, 'inp_radiance.npy'), rad_64)
	np.save(os.path.join(args.dest, 'out_cot_3d.npy'), cot_true_64)
	np.save(os.path.join(args.dest, 'out_cot_1d.npy'), cot_1d_64)
	print('\nSaved files in ', args.dest)
	print('Finished!\n')
