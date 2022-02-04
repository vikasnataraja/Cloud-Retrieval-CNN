import numpy as np
from keras.utils import Sequence, to_categorical
import os
import h5py
import cv2


def normalize_img(img):
    """ Normalizes an image by its maximum and minimum values """
    return (img-img.min())/(img.max()-img.min())


def standard_normalize(img):
    """ Normalizes an image by its mean and standard deviation.
    The mean is subtracted from each pixel and then divided by its std. deviation (only if > 0).
    """
    if img.std()!=0.:
        img = img.astype('float32')
        means = img.mean(axis=(0,1), dtype='float64')
        devs = img.std(axis=(0,1), dtype='float64')
        img = (img - means)/devs
        img = np.clip(img, -1.0, 1.0)
        img = (img + 1.0)/2.0
    return img


def resize_img(img, resize_dims, interp_method=None):
    """ Resize an image to new dimensions, uses bilinear interpolation by default. """
    if interp_method is None:
        interp_method = cv2.INTER_LINEAR # bilinear interpolation
    return cv2.resize(img, resize_dims, interp_method)


def preprocess(img, resize_dims=None, normalize=False):
    """ Pre-processing steps for an image:
    1. Normalization
    2. Resizing
    3. Reshaping to model dimensions
    """
    if normalize:
        img = standard_normalize(img)
    if resize_dims is not None:
        img = resize_img(img, resize_dims)

    if len(img.shape) > 2:
        img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
    else:
        img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
    return img


class ImageGenerator(Sequence):
    """
    Generator inherited from keras.utils.Sequence, used to feed batch-wise data to model during run.

    Adapted from sources:
    https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow/blob/master/utils/preprocessing.py
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, image_list, image_dict, label_dict,
                num_classes, batch_size, input_shape, output_shape,
                num_channels, normalize,
                to_fit, shuffle):

        self.image_list = image_list
        self.image_dict = image_dict
        self.label_dict = label_dict
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_channels = num_channels
        self.to_fit = to_fit
        self.shuffle = shuffle
        self.normalize = normalize
        if self.to_fit:
            self.on_epoch_end()

    def _data_generator(self, batch_images):
        X = np.zeros((self.batch_size, self.input_shape, self.input_shape, self.num_channels)) # input

        y = np.zeros((self.batch_size, self.output_shape, self.output_shape, self.num_classes)) # target or ground truth

        for i, image_key in enumerate(batch_images):
            """
            Performs the following steps:

            1. Read in input image from image dictionary
            2. Read in ground truth segmentation mask from label dictionary
            3. Perform pre-processing (if needed) on input image
            4. One-Hot Encode the ground truth mask
            """
            img = self.image_dict[image_key]
            label = self.label_dict[image_key]
            if self.normalize:
                img = standard_normalize(img)
            img = np.reshape(img, (img.shape[0], img.shape[1], self.num_channels))
            X[i] = img

            # one-hot encoding of mask labels using Keras. This will transform mask from
            # (width x height) to (width x height x num_classes) with 1s and 0s
            label = np.uint8(to_categorical(label, num_classes=self.num_classes))
            y[i] = label
        return X, y

    def __len__(self):
        return int(np.floor(len(self.image_list)/self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_images = [self.image_list[k] for k in indices]

        # Generate data and ground truth
        X, y = self._data_generator(batch_images)
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_list))
        if self.shuffle:
            np.random.shuffle(self.indices)


# binning for conversion from COT-space to discrete class space
cot_bins = np.concatenate((np.arange(0.0, 1.0, 0.1),
                           np.arange(1.0, 10.0, 1.0),
                           np.arange(10.0, 20.0, 2.0),
                           np.arange(20.0, 50.0, 5.0),
                           np.arange(50.0, 101.0, 10.0)))

pxvals = np.arange(0, cot_bins.shape[0])

""" Helper functions """

def get_model(path):
    import tensorflow as tf
    from utils.losses import focal_loss
    from keras.models import load_model
    return load_model('{}'.format(path), custom_objects={"tf":tf, "focal_loss":focal_loss})


def bin_cot_to_class(cot_true):
    """ Bins COT pixels to a class and returns as a mask """
    classmap = np.zeros((cot_true.shape[0], cot_true.shape[1]), dtype='uint8')
    for k in range(pxvals.size):
        if k < (pxvals.size-1):
            classmap[np.bitwise_and(cot_true>=cot_bins[k], cot_true<cot_bins[k+1])] = pxvals[k]
        else:
            classmap[cot_true>=cot_bins[k]] = pxvals[k]
    return classmap


def get_data(fdir, rad_keyname='rad_3d', cot_true_keyname='cot_true', cot_1d_keyname='cot_1d'):
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
    """ Uses HDF5 files to get radiance and COT data.
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


def get_raw_data(fdir, keyword=None, crop=False, rad_key='rad_3d', cot_true_key='cot_true', cot_1d_key='cot_1d'):
    """ Gets the raw/unformatted radianace, ground truth COT and 1D COT stored in COT space """
    store_rads, store_cot_true, store_cot_1d = {}, {}, {}
    fnames = sorted([file for file in os.listdir(fdir) if file.endswith('.h5')]) # h5 file names
    if keyword is not None:
        fnames = sorted([file for file in fnames if keyword in file])

    for i in range(len(fnames)):
        f = h5py.File(os.path.join(fdir, fnames[i]), 'r')
        if len(f.keys()) > 3: # original 7SEAS data files have different keys
            rad_key = 'rad_mca_3d' # radiance key
            cot_true_key = 'cot_inp_3d' # 3D COT ground truth
            cot_1d_key = 'cot_ret_3d' # 1D COT
            store_rads[fnames[i]] = np.array(f[rad_key][...][:, :, 0, 2], dtype='float32')
            store_cot_true[fnames[i]] = np.array(f[cot_true_key][...][:, :, 0, 2], dtype='float32')
            store_cot_1d[fnames[i]] = np.array(f[cot_1d_key][...][:, :, 0, 2], dtype='float32')
        else:
            store_rads[fnames[i]] = np.array(f[rad_key], dtype='float32')
            store_cot_true[fnames[i]] = np.array(f[cot_true_key], dtype='float32')
            store_cot_1d[fnames[i]] = np.array(f[cot_1d_key], dtype='float32')

    if crop:
        return (extract_sub_patches(store_rads, 64, 'data'),
                extract_sub_patches(store_cot_true, 64, 'data'),
                extract_sub_patches(store_cot_1d, 64, 'data'))
    return store_rads, store_cot_true, store_cot_1d
