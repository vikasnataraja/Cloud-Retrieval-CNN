import os
import argparse
import numpy as np
import h5py

# binning for conversion from COT-space to discrete class space
cot_bins = np.concatenate((np.arange(0.0, 1.0, 0.1),
                           np.arange(1.0, 10.0, 1.0),
                           np.arange(10.0, 20.0, 2.0),
                           np.arange(20.0, 50.0, 5.0),
                           np.arange(50.0, 101.0, 10.0)))

pxvals = np.arange(0, cot_bins.shape[0])


def bin_cot_to_class(cot_true):
    """ Bins COT pixels to a class and returns as a mask """
    classmap = np.zeros((cot_true.shape[0], cot_true.shape[1]), dtype='uint8')
    for k in range(pxvals.size):
        if k < (pxvals.size-1):
            classmap[np.bitwise_and(cot_true>=cot_bins[k], cot_true<cot_bins[k+1])] = pxvals[k]
        else:
            classmap[cot_true>=cot_bins[k]] = pxvals[k]
    return classmap

def get_training_data(fdir, rad_keyname='rad_3d', cot_true_keyname='cot_true', cot_1d_keyname='cot_1d'):
    """ Uses HDF5 files to get radiance and COT data.
    True COT data is discretized into a mask.
    Args:
        - fdir: (str) directory containing HDF5 files.
        - rad_keyname: (str) key for radiance data in HDF5 files.
        - cot_true_keyname: (str) key for true COT data in HDF5 files.
        - cot_1d_keyname: (str) key for 1D or IPA COT data in HDF5 files.
    Returns:
        A tuple of 3 elements:
        - store_rads: (dict) dictionary containing radiance data.
        - store_cot_true: (dict) dictionary containing true COT data as a binned mask.
        - store_cot_1d: (dict) dictionary containing IPA COT data.
    """

    fnames = sorted([file for file in os.listdir(fdir) if file.endswith('.h5')])
    store_rads, store_cot_true, store_cot_1d = {}, {}, {}

    for i in range(len(fnames)):
        f = h5py.File(os.path.join(fdir,fnames[i]), 'r')

        if len(f.keys()) > 3: # original 7 SEAS data has 6 keys
            rad_keyname = 'rad_mca_3d' # radiance key
            cot_true_keyname = 'cot_inp_3d' # 3D COT ground truth
            cot_1d_keyname = 'cot_ret_3d' # IPA COT retrieval

            cot_true = f[cot_true_keyname][...][:, :, 0, 2]
            store_cot_true[fnames[i]] = bin_cot_to_class(cot_true)

            rad = f[rad_keyname][...]
            store_rads[fnames[i]] = np.float32(rad[:, :, 0, 2])

            cot_1d = f[cot_1d_keyname][...]
            store_cot_1d[fnames[i]] = np.float32(cot_1d[:, :, 0, 2])

        else: # coarsened data has only 3 keys
            cot_true = f[cot_true_keyname]
            store_cot_true[fnames[i]] = bin_cot_to_class(cot_true)

            rad = f[rad_keyname]
            store_rads[fnames[i]] = np.float32(rad)

            cot_1d = f[cot_1d_keyname]
            store_cot_1d[fnames[i]] = np.float32(cot_1d)

    return store_rads, store_cot_true, store_cot_1d


def get_rgb_radiance_data(fdir):
    """ Uses HDF5 files to get three-channel radiance and COT data.
    True COT data is discretized into a mask.
    Args:
        - fdir: (str) directory containing HDF5 files.
    Returns:
        A tuple of 3 elements:
        - store_rads: (dict) dictionary containing radiance data.
        - store_cot_true: (dict) dictionary containing true COT data as a binned mask.
        - store_cot_1d: (dict) dictionary containing IPA COT data.
    """

    fnames = sorted([file for file in os.listdir(fdir) if file.endswith('.h5')])
    store_rads, store_cot_true, store_cot_1d = {}, {}, {}
    rad_keyname = 'rad_mca_3d' # radiance key
    cot_true_keyname = 'cot_inp_3d' # 3D COT ground truth
    cot_1d_keyname = 'cot_ret_3d' # IPA COT retrieval
    for i in range(len(fnames)):
        f = h5py.File(os.path.join(fdir,fnames[i]), 'r')
        cot_true = f[cot_true_keyname][...][:, :, 0, 2]
        store_cot_true[fnames[i]] = bin_cot_to_class(cot_true)
        
        rad = f[rad_keyname][...]
        red_channel = np.float32(rad[:, :, 0, 2])
        green_channel = np.float32(rad[:, :, 0, 1])
        blue_channel = np.float32(rad[:, :, 0, 0])
        store_rads[fnames[i]] = np.stack([red_channel, green_channel, blue_channel], axis=-1)

        cot_1d = f[cot_1d_keyname][...]
        store_cot_1d[fnames[i]] = np.float32(cot_1d[:, :, 0, 2])

    return store_rads, store_cot_true, store_cot_1d


def extract_sub_patches(img_dict, crop_dims, fname_prefix, excl_borders=0):
    """ Extracts sub-patches from an LES scene (radiance or COT).
    Currently uses 50% overlap and discards right and bottom pixels.
    Args:
        - img_dict: (dict) dictionary containing key-value pairs of original dimensions, values must be ndarray
        - crop_dims: (int) dimensions to which images must be cropped
        - fname_prefix: (str) string value to use as keynames (prefix for number)
        - excl_borders: (int) number of pixels to exclude from each of the 4 sides of an image
    Returns:
        - return_imgs: (dict), dictionary containing cropped images. The keys will be 'fname_prefix_counter'
    """
    imgs = np.array(list(img_dict.values()))
    img_names_length = len(img_dict)
    width, height = imgs.shape[1], imgs.shape[2]
    return_imgs = {}
    counter = 0
    for idx in range(img_names_length):
        for _, i in enumerate(range(0+excl_borders, width-excl_borders, int(crop_dims/2))):
            for _, j in enumerate(range(0+excl_borders, height-excl_borders, int(crop_dims/2))):
                if len(imgs.shape) > 3:
                    cropped_img = imgs[idx, i:i+crop_dims, j:j+crop_dims,:]
                else:
                    cropped_img = imgs[idx, i:i+crop_dims, j:j+crop_dims]
                if cropped_img.shape[:2] == (crop_dims, crop_dims):
                    return_imgs['{}_{}'.format(fname_prefix, counter)] = cropped_img
                    counter += 1

    print('Total number of images = {}'.format(len(return_imgs)))
    return return_imgs


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
	radiance, cot_true, cot_1d = get_training_data(args.fdir, rad_keyname='rad_mca_3d', cot_true_keyname='cot_inp_3d', cot_1d_keyname='cot_ret_3d')
	# radiance, cot_true, cot_1d = get_rgb_radiance_data(args.fdir)
	
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
