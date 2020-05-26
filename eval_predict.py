from keras.models import load_model
from utils.utils import ImageGenerator
from utils.model_utils import UpSample
from utils.losses import focal_loss
import tensorflow as tf
import os
import numpy as np
import cv2
import argparse

def preprocess(img, resize_dims, normalize):
  generator = ImageGenerator(to_fit=False)
  if normalize:
    img = generator.standard_normalize(img)
  img = generator.resize_img(img,resize_dims)
  if len(img.shape)>1:
    img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
  else:
    img = np.reshape(img,(1,img.shape[0],img.shape[1],1))
  return img

def iou(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def predict_cot(image_path, model_path, ground_truth_path):
  # load the appropriate model
  model = load_model(model_path, custom_objects={"tf":tf, "focal_loss":focal_loss})
  # read in the image
  img = cv2.imread(image_path)
  # pre-process the image and resize to model's input dimensions
  img = preprocess(img, resize_dims=(model.input_shape[1], model.input_shape[2])) 
  # make the prediction
  prediction = model.prediction(img)
  # resize to output dimensions
  prediction = np.reshape(prediction.flatten(), model.output_shape[1:])
  # use argmax to get the image
  predicted_img = np.argmax(prediction, axis=-1) 
  # write to file
  cv2.imwrite('prediction.png', predicted_img)
  if ground_truth_path is not None:
    gt = cv2.imread(ground_truth_path)
    print('IoU: {}%'.format(iou(gt, predicted_img)*100))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", default='weights/model.h5', type=str, help="Path to the trained model")
  parser.add_argument("--image_path", default=None, type=str, help="Path to the image")
  parser.add_argument("--ground_truth_path", default=None, type=str, help="Path to the ground truth image, will also calculate the iou between predicted image and the ground truth image"
  args = parser.parse_args()

  prediction(img_path=args.image_path, model_path=args.model_path, ground_truth_path=args.ground_truth_path)
