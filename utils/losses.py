import numpy as np
import tensorflow as tf
import keras.backend as K

def dice_coefficient_loss(y_true, y_pred):
  return 1-dice_coefficient(y_true, y_pred)


def dice_coefficient(y_true, y_pred, smooth=1.):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2.*intersection + smooth)/(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def jaccard_distance_loss(y_true, y_pred, smooth=100.):
  """Jaccard distance for semantic segmentation.
  Adapted from https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
  Also known as the intersection-over-union loss.
  This loss is useful when you have unbalanced numbers of pixels within an image
  because it gives all classes equal weight """
  intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
  union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
  jac = (intersection + smooth) / (union - intersection + smooth)
  return (1 - jac) * smooth


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.):
  """ Focal loss for object detection and semantic segmentation.
  Reference: Lin et al., 2018, "Focal Loss for Dense Object Detection"
  https://arxiv.org/pdf/1708.02002.pdf
  Focal loss is a modification of cross entropy loss that reduces relative loss
  for well-classified examples and focuses more on hard, misclassified samples with
  the use of a gamma paramter and a (1-p)^gamma factor. alpha is a weighting factor.
  """
  y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
  epsilon = K.epsilon()
  y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
  loss = -alpha * K.pow(1. - y_pred, gamma) * y_true * K.log(y_pred)
  return K.mean(loss, axis=-1)

def combined_loss(y_true, y_pred, focal_loss_weight=0.25):
  return focal_loss_weight*focal_loss(y_true, y_pred) + (1. - focal_loss_weight)*focal_tversky_loss(y_true, y_pred)


