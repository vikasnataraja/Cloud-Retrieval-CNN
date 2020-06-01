from keras.models import load_model
from utils.utils import standard_normalize, resize_img
# from utils.model_utils import UpSample
from utils.losses import focal_loss
from utils.eval_metrics import plot_evaluation
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


def predict_cot_on_image(img, model, ground_truth_img=None):
  """
  Predict the COT for an image (numpy array) and if ground truth is available, evaluate model performance.
  """
  # pre-process the image and resize to model's input dimensions
  img = preprocess(img, resize_dims=(model.input_shape[1], model.input_shape[2])) 

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
    plot_evaluation(ground_truth, prediction)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", default='weights/model.h5', type=str, help="Path to the trained model")
  parser.add_argument("--image_path", default=None, type=str, help="Path to the image")
  parser.add_argument("--ground_truth_path", default=None, type=str, help="Path to the ground truth image, will also calculate the iou between predicted image and the ground truth image")
  args = parser.parse_args()

  model = load_model(args.model_path, custom_objects={"tf":tf, "focal_loss":focal_loss})
  img = plt.imread(args.image_path)
  if args.ground_truth_path is not None:
    ground_truth = plt.imread(args.ground_truth_path)
  else:
    ground_truth = None

  prediction(img, model, ground_truth)

