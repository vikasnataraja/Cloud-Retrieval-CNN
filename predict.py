import numpy as np
import os
import argparse
import h5py

cot_class_l = np.concatenate((np.arange(0.0, 1.0, 0.1),
                               np.arange(1.0, 10.0, 1.0),
                               np.arange(10.0, 20.0, 2.0),
                               np.arange(20.0, 50.0, 5.0),
                               np.arange(50.0, 101.0, 10.0)))

# COT classes (use upper boundary)
cot_class_r = np.append(cot_class_l[1:], 200.0)

# COT classes (use middle)
cot_class_mid = (cot_class_l+cot_class_r)/2.0

def post_process(probs, width=5, low_limit=0, up_limit=35):
    """
    probs: arr, 64x64x36, probabilties between 0 and 1
    """
    dir_width = int((width-1)/2) # directional width
    probs_2d = probs.reshape((probs.shape[-3]*probs.shape[-2], probs.shape[-1])) # flattened cuboid
    result = np.zeros(probs.shape[:-1]).ravel() # 64x64 array
    max_class_idxs = np.argmax(probs_2d, axis=-1) # identify class with max.probability for each pixel
    z_idxs = np.where(max_class_idxs==0)[0]
    result[z_idxs] = np.dot(probs_2d[z_idxs, 0:dir_width+1], cot_class_mid[0:dir_width+1])
    nz_idxs = np.where(max_class_idxs>0)[0]
    for idx in nz_idxs:
        max_class = max_class_idxs[idx]
        wdw_classes = np.array(list(set(np.clip(list(range(max_class-dir_width, max_class+dir_width+1)), low_limit, up_limit))))
        result[idx] = np.dot(probs_2d[idx][wdw_classes], cot_class_mid[wdw_classes])
    return result.reshape((probs.shape[:-1]))


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


def predict_all_at_once(data, model):
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
    from utils.plot_utils import prediction_panel_viz
    from utils.utils import get_model
    
    input_data = np.reshape(rad, (1, 64, 64, 1)) # model expects 4D tensor
    model = get_model(modelpath)
    probs = model.predict(input_data, verbose=1) # probabilities predicted by CNN, of shape (1, 64, 64, 36)
    # cot_cnn = np.dot(probs, cot_class_mid).reshape((64, 64)) # predicted COT of shape (64, 64)
    cot_cnn = post_process(probs[0]) # window-method post-processing i.e gaussian window
    print('Prediction finished, plotting ...')
    prediction_panel_viz(rad, cot_true, cot_1d, cot_cnn)
    print('Finished!')

fname = 'data/test_data/00002122_0.0024_0.0187_[00000122](128-192_128-192)_data_x48km_TB_nt035_undg_tau1h_nndg_tau1h_v03_shear_coa-fac-1_coa-fac-1_600nm.h5'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str, help="Path to the model")
    args = parser.parse_args()

    test_model(fname, args.model_path)
