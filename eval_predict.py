from keras.models import load_model
from utils.utils import standard_normalize, resize_img
# from utils.model_utils import UpSample
from utils.losses import focal_loss, combined_loss
from utils.eval_utils import plot_evaluation, visualize_prediction, plot_stat_metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import numpy as np
import argparse
import cv2

def preprocess(img, resize_dims=None, normalize=False):
  
  if normalize:
    img = standard_normalize(img)
  if resize_dims is not None:
    img = resize_img(img,resize_dims)

  if len(img.shape)>2:
    img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
  else:
    img = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
  return img


def predict_on_validation_set(input_data, gt_data, model, validation_list=None):
  """
  Predict for a set of validation images or samples and visualize the peformance of the model statistically.
  Args:
      - input_data: dict, dictionary that contains the input data
      - gt_data: dict, dictionary that contains the ground truth data with same keys as `input_data`
      - model: keras.models.Model object, the model loaded from keras
      - validation_list: list, a python list containing the keys to the validation set like ['data_10', 'data_100'...].
			 If None, then all the keys from the original input_data will be used instead.

  Returns:
      - means: list, python list of means of each input image
      - devs: list, python list of standard deviations from mean of each input image
      - slopes: list, python list of slopes calculated by np.mean((non-zero predictions)/(non-zero ground truths))

  """
  devs, means, slopes = [], [], []
  if validation_list is None:
    validation_list = list(input_data.keys())

  for randkey in validation_list:
    input_img = input_data[randkey]
    gt_img = gt_data[randkey]
    img = preprocess(input_img)

    temp = model.predict(img)
    temp = np.reshape(temp.flatten(), model.output_shape[1:])
    prediction = np.argmax(temp, axis=-1)

    flat_pred = prediction.flatten()
    flat_gt = gt_img.flatten()

    non_zero_idx = np.where(flat_gt>0)[0] # indices that have non-zero classes
    non_zero_gt = flat_gt[non_zero_idx]
    non_zero_prediction = flat_pred[non_zero_idx]
    if non_zero_prediction.shape[0]==0: # break current iteration if all classes are zero
      continue

    slope = non_zero_prediction/non_zero_gt # slope is element-wise division of non-zero classes

    means.append(np.mean(input_img))
    devs.append(np.std(input_img))
    slopes.append(np.mean(slope))
  
  plot_stat_metrics(means, devs, slopes)

  return means, devs, slopes


def predict_on_random_validation_image(input_data, gt_data, model, keyname=None):
  """
  Predict COT for a random validation image and visualize it.
  Args:
      - input_data: dict, dictionary that contains the input data.
      - gt_data: dict, dictionary that contains the ground truth data with same keys as `input_data`.
      - model: keras.models.Model object, the model loaded from keras.
      - keyname: str, the specific image's key in the dictionaries of input_data and gt_data to visualize that image.
                By default, a random validation image is visualized and therefore this arg is set to None.
                Pass a string like 'data_977' to visualize the plots for a particular image.

  Returns:
      - prediction: arr, a numpy array with the same shape as input image.

  """
  
  if keyname is not None:
    random_img_key = keyname
  else:
    # split to training and validation
    X_train, X_val = train_test_split(list(input_data.keys()),shuffle=True, random_state=42, test_size=0.20)
    random_img_key = np.random.choice(X_val)

  input_img = input_data['{}'.format(random_img_key)]
  gt_img = gt_data['{}'.format(random_img_key)]

  # pre-process the image and resize to model's input dimensions
  img = preprocess(input_img, resize_dims=(model.input_shape[1], model.input_shape[2])) 

  # make the prediction
  temp = model.predict(img)
  
  # resize to output dimensions
  temp = np.reshape(temp.flatten(), model.output_shape[1:])
  
  # use argmax to get the image
  prediction = np.argmax(temp, axis=-1)

  print('Visualizing image {}:\n'.format(random_img_key))
  visualize_prediction(input_img, gt_img, prediction)
  plot_evaluation(gt_img, prediction)
  
  return prediction


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
  # pre-process the image and resize to model's input dimensions
  img = preprocess(input_img, resize_dims=(model.input_shape[1], model.input_shape[2])) 

  # make the prediction
  temp = model.predict(img)
  
  # resize to output dimensions
  temp = np.reshape(temp.flatten(), model.output_shape[1:])
  
  # use argmax to get the image
  prediction = np.argmax(temp, axis=-1) 
  
  # write to file
  cv2.imwrite('results/prediction.png', prediction)
  print('Saved predicted image in "results/" as "prediction.png"')
  
  if ground_truth_img is not None:
    visualize_prediction(input_img, ground_truth_img, prediction)
    plot_evaluation(ground_truth_img, prediction)

  return prediction


if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", default='weights/model.h5', type=str, help="Path to the trained model")
  parser.add_argument("--image_path", default=None, type=str, help="Path to the image")
  parser.add_argument("--ground_truth_path", default=None, type=str, help="Path to the ground truth image, will also calculate the iou between predicted image and the ground truth image")
  parser.add_argument("--input_file", default=None, type=str, help="Path to the npy file containing input data")
  parser.add_argument("--output_file", default=None, type=str, help="Path to the npy file containing ground truth data")
  parser.add_argument("--keyname", default=None, type=str, help="Key of the specific image to display among the validation image. For example, 'data_977', 'data_34' etc. By default, this is set to None to use a random validation image")

  args = parser.parse_args()

  model = load_model(args.model_path, custom_objects={"tf":tf, "focal_loss":focal_loss})
  
  if args.image_path is not None: # predict for a given image
    img = plt.imread(args.image_path)
    if args.ground_truth_path is not None:
      ground_truth = plt.imread(args.ground_truth_path)
    else:
      ground_truth = None
 
    prediction = predict_cot_on_image(img, model, ground_truth)
  
  else: # predict on validation
    in_data = np.load(args.input_file, allow_pickle=True).item()
    out_data = np.load(args.output_file, allow_pickle=True).item()
    prediction = predict_random_validation_image(in_data, out_data, model, args.keyname)


