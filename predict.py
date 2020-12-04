from utils.utils import preprocess, get_class_space_data, get_cot_space_data, get_model
# from utils.model_utils import UpSample
from utils.plot_utils import plot_model_comparison, plot_all
from time import perf_counter
import tensorflow as tf
import numpy as np
import os
import argparse
import cv2
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

def get_progress(start, total, current_count):
  """
  Get progress of prediction for large files and print to stdout.
  
  Args:
    - start: float, time at which operation started.
    - total: int, total number of iterations in the loop.
    - current_count: int, current iteration.
  """

  if current_count == 0: current_count = 1 # avoid division by zero
  stop = perf_counter()
  remaining = round((stop - start) * ((total/current_count) - 1))
  progress = 100 * current_count / total
  if current_count % np.floor(total/5) == 0: # print at 20, 40, 60, 80% progress
      if remaining > 60: # print in min 
          print('Progress: {:.0f}%, ~ {:0.1f} min remaining'.format(np.ceil(progress), remaining/60))
      else:  # print in seconds
          print('Progress: {:.0f}%, ~ {}s remaining'.format(np.ceil(progress), remaining))


def predict_on_validation_set(input_data, gt_data, model, image_list=None):
  """
  Predict for a set of validation images or samples and visualize the peformance of the model statistically.
  Args:
    - input_data: dict, dictionary that contains the input data
    - gt_data: dict, dictionary that contains the ground truth data with same keys as `input_data`
    - model: keras.models.Model object, the model loaded from keras
    - image_list: list, a python list containing the keys to a subset set like ['data_10', 'data_100'...].
                  If None, then all the keys from the original input_data will be used.

  Returns:
    - means: list, python list of means of each input image
    - devs: list, python list of standard deviations from mean of each input image
    - slopes: list, python list of slopes calculated by np.mean((non-zero predictions)/(non-zero ground truths))

  """
  devs, means, slopes = [], [], []
  if image_list is None:
    image_list = list(input_data.keys())
  
  print('Starting evaluation on {} images'.format(len(image_list)))
  start = perf_counter() # start timer
  for count, randkey in enumerate(image_list):
    input_img = input_data[randkey]

    temp = model.predict(preprocess(input_img))
    temp = np.reshape(temp.ravel(), model.output_shape[1:])
    prediction = np.argmax(temp, axis=-1)

    flat_pred = prediction.ravel()
    flat_gt = gt_data[randkey].ravel()

    non_zero_idx = np.where(flat_gt > 0)[0] # indices that have non-zero classes
    non_zero_gt = flat_gt[non_zero_idx]
    non_zero_prediction = flat_pred[non_zero_idx]
    if non_zero_prediction.shape[0] != 0: # if all classes are zero, don't add to list
      slope = non_zero_prediction/non_zero_gt # slope is element-wise division of non-zero classes
      means.append(np.mean(input_img))
      devs.append(np.std(input_img))
      slopes.append(np.mean(slope))
    
    get_progress(start, len(image_list), count)

  return means, devs, slopes


def get_1d_retrievals(input_data, cot_1d, cot_3d, image_list=None):
  """ Retrieve the 1D retrievals from the data and calculate the slope, mean and std. dev """

  if image_list is None:
    image_list = list(input_data.keys())
  slopes1d, devs1d, means1d = [], [], []
  for i in image_list:
    rad = input_data[i]
    flat_pred = cot_1d[i].ravel()
    flat_gt = cot_3d[i].ravel()

    non_zero_idx = np.where(flat_gt > 0)[0] # indices that have non-zero classes
    non_zero_gt = flat_gt[non_zero_idx]
    non_zero_prediction = flat_pred[non_zero_idx]
    if non_zero_prediction.shape[0] != 0:
      slope = non_zero_prediction/non_zero_gt
      slopes1d.append(np.mean(slope))
      devs1d.append(np.std(rad))
      means1d.append(np.mean(rad))

  return means1d, devs1d, slopes1d


def predict_cot_on_image(input_img, model, ground_truth_img=None):
  """
  Predict the COT for an image (numpy array) and if ground truth is available, evaluate model performance.
  Args:
      - input_img: arr, a numpy array representing the input image for the model.
      - model: keras.models.Model object, the model loaded using Keras.
      - ground_truth_img: arr, a numpy array representing the ground truth image. If None, only the prediction
                          on the image will be saved to file without plots.

  Returns:
      - prediction: arr, a numpy array with the same shape as input_img.
  """
  # make the prediction
  temp = model.predict(preprocess(input_img, resize_dims=(model.input_shape[1], model.input_shape[2])))
  
  # resize to output dimensions
  temp = np.reshape(temp.ravel(), model.output_shape[1:])
  
  # use argmax to get the image
  prediction = np.argmax(temp, axis=-1) 
  
  # write to file
  cv2.imwrite('results/prediction.png', prediction)
  print('Saved predicted image in "results/" as "prediction.png"')
  
  if ground_truth_img is not None:
    visualize_prediction(input_img, ground_truth_img, prediction)
    plot_evaluation(ground_truth_img, prediction)

  return prediction


def predict_on_dataset(input_data, model, use_argmax=False):
  # ====================================================== #
  def predict_on_image(input_img, model, use_argmax):
    temp = model.predict(preprocess(input_img, resize_dims=(model.input_shape[1], model.input_shape[2])))
    temp = np.reshape(temp.ravel(), model.output_shape[1:])
    if use_argmax:
      return np.argmax(temp, axis=-1) # use argmax to get the image
    return np.dot(temp, cot_class_mid)
  # ====================================================== #
  predictions = {}
  for idx, key in enumerate(list(input_data.keys())):
    input_img = input_data[key]
    predictions['data_{}'.format(idx)] = predict_on_image(input_img, model, use_argmax=use_argmax)
  return predictions


def reconstruct(dictionary, num_scenes, dims): 
  keys = list(dictionary.keys())
  recon = np.zeros((num_scenes, dims-32, dims-32)) 
  idx = 0
  for up_idx in range(num_scenes):
    for i in range(0, dims-32, 32):
      for j in range(0, dims-32, 32):
        recon[up_idx, i:i+32, j:j+32] = dictionary[keys[idx]][:32,:32]
        idx += 1                    
  return recon


def compare_models(input_file, file_1d, file_3d, modelpath_1, modelpath_2, compare, figname):
  radiances = np.load('{}'.format(input_file), allow_pickle=True).item()
  cot_1d = np.load('{}'.format(file_1d), allow_pickle=True).item()
  cot_3d = np.load('{}'.format(file_3d), allow_pickle=True).item()
  
  model_1 = get_model(modelpath_1)
  means3d_1, devs3d_1, slopes3d_1 = predict_on_validation_set(radiances, cot_3d, model_1)

  means1d, devs1d, slopes1d = get_1d_retrievals(radiances, cot_1d=cot_1d, cot_3d=cot_3d)

  if not compare:
    plot_model_comparison(means1d, devs1d, slopes1d, slopes3d_1, label1='1d_ret', label2='3d_ret', figname=figname)

  else:
    model_2 = get_model(modelpath_2)
    means3d_2, devs3d_2, slopes3d_2 = predict_on_validation_set(radiances, cot_3d, model_2)
    plot_model_comparison(means1d, devs1d, slopes3d_1, slopes3d_2, label1='model_1', label2='model_2', figname=figname)
  
  print('The mean slope of 1D retrievals is {:0.2f}\n'.format(np.mean(slopes1d)))
  print('\nThe mean slope of 3D retrievals for model_1 is {:0.2f}\n'.format(np.mean(slopes3d_1)))
  print('\nThe mean slope of 3D retrievals for model_2 is {:0.2f}\n'.format(np.mean(slopes3d_2)))


def predict_metrics(path_to_model, fdir, input_file, file_1d, file_3d, reconstruct, figname):

  fnames = [file for file in os.listdir(fdir) if file.endswith('.h5')]

  rad_cot_space, cot_true_cot_space, cot_1d_cot_space = get_cot_space_data(fdir, fnames)
  _, cot_true_class_space, cot_1d_class_space = get_class_space_data(input_file, file_3d, file_1d)
  model = get_model(path_to_model)
  prediction_cot_space = predict_on_dataset(rad_cot_space, model, use_argmax=False)
  prediction_class_space = predict_on_dataset(rad_cot_space, model, use_argmax=True)
  
  if reconstruct:
    recon_true_class_space = reconstruct(cot_true_class_space, num_scenes=5, dims=384)
    recon_true_cot_space = reconstruct(cot_true_cot_space, num_scenes=5, dims=384)

    recon_pred_1d_class_space = reconstruct(cot_1d_class_space, num_scenes=5, dims=384)
    recon_pred_1d_cot_space = reconstruct(cot_1d_cot_space, num_scenes=5, dims=384)

    recon_pred_cnn_class_space = reconstruct(prediction_class_space, num_scenes=5, dims=384)
    recon_pred_cnn_cot_space = reconstruct(prediction_cot_space, num_scenes=5, dims=384)

    recon_input_radiance = reconstruct(rad_class_space, num_scenes=5, dims=384)
    plot_all(recon_input_radiance, recon_true_cot_space, recon_true_class_space,
         recon_pred_cnn_cot_space, recon_pred_cnn_class_space,
         recon_pred_1d_cot_space, recon_pred_1d_class_space,
         rows=len(recon_input_radiance), random=False, dimensions='480x480', figsize=(42,38))
  else:
    plot_all(rad_cot_space, cot_true_cot_space, cot_true_class_space, 
           prediction_cot_space, prediction_class_space,
           cot_1d_cot_space, cot_1d_class_space, filename=figname, rows=3, dimensions='64x64', random=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5dir', default=None, type=str, help="Path to the directory containing h5 files")
    parser.add_argument('--input_file', default=None, type=str,help="Path to numpy input images file")
    parser.add_argument('--file_1d', default=None, type=str, help="Path to the 1D retrievals numpy file")
    parser.add_argument('--file_3d', default=None, type=str, help="Path to the 3D retrievals numpy file")
    parser.add_argument('--model_1_path', default=None, type=str, help="Path to the first model")
    parser.add_argument('--model_2_path', default=None, type=str, help="Path to the second model, optional")
    parser.add_argument('--save_figure_as', default='figure.png', type=str, help="Filename for saving figure")
    parser.add_argument('--compare_models', dest='compare', action='store_true',
                      help="Pass --compare_models to compare two models. By default, only one model is used")
    parser.add_argument('--metrics', dest='metrics', action='store_true',
                      help="Pass --metrics to plot model evaluation with all metrics")
    parser.add_argument('--reconstruct', dest='reconstruct', action='store_true',
                      help="Pass --reconstruct to plot evaluation for reconstructed scene")
    args = parser.parse_args()
    if args.compare:
      compare_models(args.input_file, args.file_1d, args.file_3d,
                     args.model_1_path, args.model_2_path, 
                     args.compare, figname=args.save_figure_as)

    if args.metrics:
      predict_metrics(args.model_1_path, args.h5dir, args.input_file,
                      args.file_1d, args.file_3d, args.reconstruct, args.save_figure_as)


