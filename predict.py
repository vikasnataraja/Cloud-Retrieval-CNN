from utils.utils import preprocess, get_class_space_data, get_cot_space_data, get_model
from utils.plot_utils import plot_model_comparison, plot_all_metrics, plot_slopes, plot_heatmap
from time import perf_counter
import numpy as np
import os
import argparse

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
  Returns:
    Prints to stdout.
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


def get_slopes(radiance, cot_true, cot_pred, cot_1d, thresh, recon=False):
  """ Get fidelity slope metric for data.
  
  Args:
    - radiance: dict, dictionary containing radiance data.
    - cot_true: dict, ground truth dictionary.
    - cot_1d: dict, 1D retrieval data dictionary.
    - thresh: float, threshold to prevent zero division above which slopes will be calculated.
    - recon: bool, reconstruction flag, set to True if input is reconstructed scenes.
  
  Returns:
    - rad_means: list of mean values of radiance.
    - rad_stds: list of standard deviation values of radiance.
    - slopes_cnn: list of fidelity values based on CNN prediction.
    - slopes_1d: list of fidelity values based on 1D retrievals.
  """

  rad_means, rad_stds, slopes_cnn, slopes_1d = [], [], [], []
  if recon is True:
    iter = radiance.shape[0]
  for key in range(radiance.shape[0]):
    input_img = radiance[key]
    truth_cot = cot_true[key].ravel()
    ret_1d = cot_1d[key].ravel()
    pred_cot = cot_pred[key].ravel()
        
    non_zero_idx = np.where(truth_cot > thresh)[0] # indices that have pixels > thresh to avoid zero division
    non_zero_truth = truth_cot[non_zero_idx]
    non_zero_prediction = pred_cot[non_zero_idx]
    non_zero_1d = ret_1d[non_zero_idx]
    if non_zero_prediction.shape[0] != 0: # confirm that it is not an empty image
      rad_stds.append(np.std(input_img))
      rad_means.append(np.mean(input_img))
      slopes_cnn.append(np.mean(non_zero_prediction/non_zero_truth))
      slopes_1d.append(np.mean(non_zero_1d/non_zero_truth))
    return rad_means, rad_stds, slopes_cnn, slopes_1d


def get_1d_retrievals(input_data, cot_1d, cot_3d):
  """ Get the 1D retrievals from the data and calculate the slope(fidelity), mean and std. dev 
  Args:
    - input_data: dict, dictionary containing radiance data.
    - cot_1d: dict, dictionary containing 1D COT retrieval data.
    - cot_3d: dict, dictionary containing 3D COT retrieval data.
  Returns:
    - means1d: list, list of mean value of each radiance image in `input_data`.
    - devs1d: list, list of standard deviation value of each radiance image in `input_data`.
    - slopes1d: list, list of slopes/fidelity calculated values for each retrieval.
  """

  slopes1d, devs1d, means1d = [], [], []
  for i in list(input_data.keys()):
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


def predict_on_single_image(img, model, use_argmax):
  """ Predict on a single image with argmax or weighted means 
  Args:
    - img: arr, input image of size (model.input_shape[1] x model.input_shape[2]).
    - model: trained model.
    - use_argmax: bool, set to True to use argmax method, False to use weighted means method.
  Returns:
    - predicted COT image of size (model.output_shape[1] x model.output_shape[2]).
  """
  temp = model.predict(preprocess(img, resize_dims=(model.input_shape[1], model.input_shape[2])))
  temp = np.reshape(temp.ravel(), model.output_shape[1:])
  if use_argmax:
    return np.argmax(temp, axis=-1) # use argmax to get the image
  return np.dot(temp, cot_class_mid)


def predict_on_dataset(input_data, model, use_argmax=False):
  """ Predict on dictionary of input data using model """

  predictions = {}
  start = perf_counter() # start timer
  for idx, key in enumerate(list(input_data.keys())):
    input_img = input_data[key]
    predictions['data_{}'.format(idx)] = predict_on_single_image(input_img, model, use_argmax=use_argmax)
    get_progress(start, len(input_data), idx) # report progress
  return predictions


def reconstruct_scenes(data_dictionary, dims):
  """ Reconstruct sub-scenes of 64x64 back to original dimensions"""
  
  keys = list(data_dictionary.keys())
  if dims == 384:
    num_scenes = int(len(keys)/(11*11))
  else:
    num_scenes = int(len(keys)/(14*14))
  print('Total reconstructed scenes:', num_scenes)
  recon = np.zeros((num_scenes, dims-32, dims-32)) 
  idx = 0
  for up_idx in range(num_scenes):
    for i in range(0, dims-32, 32):
      for j in range(0, dims-32, 32):
        recon[up_idx, i:i+32, j:j+32] = data_dictionary[keys[idx]][:32, :32]
        idx += 1
  return recon


def plot_heatmap_slopes(path_to_model, fdir, input_file, file_1d, file_3d, dims, figname):
  """ Plot a figure with heatmap and metrics to evaluate model """
  
  rad_cot_space, cot_true_cot_space, cot_1d_cot_space = get_cot_space_data(fdir)
  rad_class_space, cot_true_class_space, cot_1d_class_space = get_class_space_data(input_file,
                                                                                   file_3d, 
                                                                                   file_1d)
  model = get_model(path_to_model)
  prediction_cot_space = predict_on_dataset(rad_cot_space, model, use_argmax=False) # use weighted means to get cot space predictions
  prediction_class_space = predict_on_dataset(rad_class_space, model, use_argmax=True) # use argmax to get class space predictions
  
  recon_true_class_space = reconstruct_scenes(cot_true_class_space, dims)
  recon_true_cot_space = reconstruct_scenes(cot_true_cot_space, dims)

  recon_pred_1d_class_space = reconstruct_scenes(cot_1d_class_space, dims)
  recon_pred_1d_cot_space = reconstruct_scenes(cot_1d_cot_space, dims)

  recon_pred_cnn_class_space = reconstruct_scenes(prediction_class_space, dims)
  recon_pred_cnn_cot_space = reconstruct_scenes(prediction_cot_space, dims)

  recon_input_radiance = reconstruct_scenes(rad_cot_space, dims)
  
  rad_means_cot, rad_stds_cot, slopes_cnn_cot_space, slopes_1d_cot_space = get_slopes(recon_input_radiance,
                                                       recon_true_cot_space,
                                                       recon_pred_cnn_cot_space,
                                                       recon_pred_1d_cot_space, thresh=0.5)

  rad_means_class, _, slopes_cnn_class_space, slopes_1d_class_space = get_slopes(recon_input_radiance, 
                                                       recon_true_class_space, 
                                                       recon_pred_cnn_class_space, 
                                                       recon_pred_1d_class_space, thresh=0)
  plot_slopes(rad_means_class, rad_means_cot,
            slopes_cnn_class_space, slopes_1d_class_space,
            slopes_cnn_cot_space, slopes_1d_cot_space,
            recon_true_class_space, recon_pred_cnn_class_space, recon_pred_1d_class_space,
            recon_true_cot_space, recon_pred_cnn_cot_space, recon_pred_1d_cot_space, filename=figname, recon=True)
  
  plot_heatmap(recon_input_radiance, recon_true_cot_space, recon_true_class_space,
         recon_pred_cnn_cot_space, recon_pred_cnn_class_space,
         recon_pred_1d_cot_space, recon_pred_1d_class_space,
         rows=6, filename=figname, random=False, dimensions='480x480', figsize=(44,42))


def predict_with_metrics(path_to_model, fdir, input_file, file_1d, file_3d, reconstruct, dims, figname):
  """ Predict on data and plot evaluation figures """

  rad_cot_space, cot_true_cot_space, cot_1d_cot_space = get_cot_space_data(fdir)
  _, cot_true_class_space, cot_1d_class_space = get_class_space_data(input_file, file_3d, file_1d)
  model = get_model(path_to_model)
  prediction_cot_space = predict_on_dataset(rad_cot_space, model, use_argmax=False) #use weighted means to get cot space predictions
  prediction_class_space = predict_on_dataset(rad_cot_space, model, use_argmax=True) #use argmax to get class space predictions
  
  if reconstruct:
    recon_true_class_space = reconstruct_scenes(cot_true_class_space, dims)
    recon_true_cot_space = reconstruct_scenes(cot_true_cot_space, dims)

    recon_pred_1d_class_space = reconstruct_scenes(cot_1d_class_space, dims)
    recon_pred_1d_cot_space = reconstruct_scenes(cot_1d_cot_space, dims)

    recon_pred_cnn_class_space = reconstruct_scenes(prediction_class_space, dims)
    recon_pred_cnn_cot_space = reconstruct_scenes(prediction_cot_space, dims)

    recon_input_radiance = reconstruct_scenes(rad_cot_space, dims)
    plot_all_metrics(recon_input_radiance, recon_true_cot_space, recon_true_class_space,
         recon_pred_cnn_cot_space, recon_pred_cnn_class_space,
         recon_pred_1d_cot_space, recon_pred_1d_class_space,
         rows=len(recon_input_radiance), random=False, filename=figname, dimensions='480x480', figsize=(42,38))
  else:
    plot_all_metrics(rad_cot_space, cot_true_cot_space, cot_true_class_space, 
           prediction_cot_space, prediction_class_space,
           cot_1d_cot_space, cot_1d_class_space, filename=figname, rows=3, dimensions='64x64', random=True)
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--h5dir', default=None, type=str, help="Path to the directory containing h5 files")
  parser.add_argument('--input_file', default=None, type=str,help="Path to numpy input images file")
  parser.add_argument('--file_1d', default=None, type=str, help="Path to the 1D retrievals numpy file")
  parser.add_argument('--file_3d', default=None, type=str, help="Path to the 3D retrievals numpy file")
  parser.add_argument('--dims', default=480, type=int, help="Dimensions of the original scenes")
  parser.add_argument('--model_1_path', default=None, type=str, help="Path to the first model")
  parser.add_argument('--model_2_path', default=None, type=str, help="Path to the second model, optional")
  parser.add_argument('--save_figure_as', default='figure.png', type=str, help="Filename for saving figure")
  parser.add_argument('--metrics', dest='metrics', action='store_true',
                      help="Pass --metrics to plot model evaluation with all metrics")
  parser.add_argument('--heatmap', dest='heatmap', action='store_true',
                      help="Pass --heatmap to plot heatmap for scene")
  parser.add_argument('--reconstruct', dest='reconstruct', action='store_true',
                      help="Pass --reconstruct to plot evaluation for reconstructed scene")
  args = parser.parse_args()

  if args.metrics:
    predict_with_metrics(args.model_1_path, args.h5dir, args.input_file,
                         args.file_1d, args.file_3d, args.reconstruct, args.dims, args.save_figure_as)

  if args.heatmap:
    plot_heatmap_slopes(args.model_1_path, args.h5dir, args.input_file,
                        args.file_1d, args.file_3d, args.dims, args.save_figure_as)


