from utils.utils import preprocess, get_model
import numpy as np
import os
import argparse
import h5py
from utils.plot_utils import prediction_panel_viz

cot_class_l = np.concatenate((np.arange(0.0, 1.0, 0.1),
                               np.arange(1.0, 10.0, 1.0),
                               np.arange(10.0, 20.0, 2.0),
                               np.arange(20.0, 50.0, 5.0),
                               np.arange(50.0, 101.0, 10.0)))

# COT classes (use upper boundary)
cot_class_r = np.append(cot_class_l[1:], 200.0)

# COT classes (use middle)
cot_class_mid = (cot_class_l+cot_class_r)/2.0


def predict_on_single_image(img, model, use_argmax):
    """ Predict on a single image with argmax or weighted means.
    Args:
        - img: (ndarray), input image of size (model.input_shape[1], model.input_shape[2]).
        - model: (keras.Model) keras model object.
        - use_argmax: (bool), set to True to use argmax method, False to use weighted means method.
    Returns:
        - (ndarray) predicted COT image of shape (model.output_shape[1], model.output_shape[2]).
    """
    temp = model.predict(preprocess(img, resize_dims=(model.input_shape[1], model.input_shape[2])))
    temp = np.reshape(temp.ravel(), model.output_shape[1:])
    if use_argmax:
        return np.argmax(temp, axis=-1) # use argmax to get the image
    return np.dot(temp, cot_class_mid)


def predict_at_once(data, model):
    """
    Predict on an array of radiance images using the weighted means method. Returns an array of the same length as `data`.

    Args:
        - data: (ndarray) a 4D array of radiance images, of shape (N, width, length, depth)
        - model: (keras.Model) the keras model object.
    Returns:
        - (ndarray) an array of shape (N, model.output_shape[1], model.output_shape[2])
    """

    if data.shape[1:] != model.input_shape[1:]:
        raise ValueError("Mismatched shapes, elements of input data are of shape {} but model expects {}".format(data.shape[1:], model.input_shape[1:]))
    return np.dot(model.predict(data, verbose=1, batch_size=32), cot_class_mid)


def test_model(filepath, modelpath):
    f = h5py.File(filepath, 'r')
    rad = np.array(f['rad_3d'], dtype='float32')
    cot_true = np.array(f['cot_true'], dtype='float32')
    cot_1d = np.array(f['cot_1d'], dtype='float32')

    input_data = np.reshape(rad, (1, 64, 64, 1)) # model expects 4D tensor
    model = get_model(modelpath)
    probs = model.predict(input_data) # probabilities predicted by CNN, of shape (1, 64, 64, 36)
    cot_cnn = np.dot(probs, cot_class_mid).reshape((64, 64)) # predicted COT of shape (64, 64)
    print('Prediction finished, plotting ...')
    prediction_panel_viz(rad, cot_true, cot_1d, cot_cnn)
    print('Finished!')

fname = 'data/test_data_sulu_sea/00010547_0.0058_0.0190_[00000547](192-256_288-352)_data_x48km_TB_nt230_undg_tau1h_nndg_tau1h_v03_shear_coa-fac-1_coa-fac-1_600nm.h5'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str, help="Path to the model")
    args = parser.parse_args()

    test_model(fname, args.model_path)
