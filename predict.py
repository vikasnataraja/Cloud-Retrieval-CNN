from keras.models import load_model
from utils.utils import standard_normalize, resize_img
# from utils.model_utils import UpSample
from utils.losses import focal_loss
from utils.plot_utils import plot_evaluation, visualize_prediction, plot_stat_metrics, plot_1d_3d
from sklearn.model_selection import train_test_split
from time import perf_counter
import tensorflow as tf
import numpy as np
import os
import argparse
import cv2

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


def preprocess(img, resize_dims=None, normalize=False):
  """ Pre-processing steps for an image """
  if normalize:
    img = standard_normalize(img)
  if resize_dims is not None:
    img = resize_img(img, resize_dims)

  if len(img.shape) > 2:
    img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
  else:
    img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
  return img


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


def main(input_file, file_1d, file_3d, modelpath):
    radiances = np.load('{}'.format(input_file), allow_pickle=True).item()
    cot_1d = np.load('{}'.format(file_1d), allow_pickle=True).item()
    cot_3d = np.load('{}'.format(file_3d), allow_pickle=True).item()

    model = load_model('{}'.format(modelpath), custom_objects={"tf":tf, "focal_loss":focal_loss})

    means3d, devs3d, slopes3d = predict_on_validation_set(radiances, cot_3d, model)
    print('\nThe mean slope of 3D retrievals is {}\n'.format(np.mean(slopes3d)))

    means1d, devs1d, slopes1d = get_1d_retrievals(radiances, cot_1d=cot_1d, cot_3d=cot_3d)
    print('The mean slope of 1D retrievals is {}\n'.format(np.mean(slopes1d)))

    plot_1d_3d(means1d, devs1d, slopes1d, slopes3d, label1='1d_ret', label2='3d_ret')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='data/single_channel/input_radiance.npy', type=str,
                        help="Path to numpy input images file")
    parser.add_argument('--file_1d', default=None, type=str, help="Path to the 1D retrievals numpy file")
    parser.add_argument('--file_3d', default=None, type=str, help="Path to the 3D retrievals numpy file")
    parser.add_argument('--model_path', default=None, type=str, help="Path to the model")
    args = parser.parse_args()

    main(args.input_file, args.file_1d, args.file_3d, args.model_path)

