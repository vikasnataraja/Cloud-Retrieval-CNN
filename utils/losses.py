import numpy as np
import tensorflow as tf
import keras.backend as K


"""
Dice coefficient and corresponding loss.
Form of IoU loss
"""
def dice_coefficient_loss(y_true, y_pred):
  return 1-dice_coefficient(y_true, y_pred)

def dice_coefficient(y_true, y_pred, smooth=1.):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2.*intersection + smooth)/(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


"""
Jaccard Distance Loss
Adapted from https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
"""
def jaccard_distance_loss(y_true, y_pred, smooth=100.):
  """Jaccard distance for semantic segmentation.
  Also known as the intersection-over-union loss.
  This loss is useful when you have unbalanced numbers of pixels within an image
  because it gives all classes equal weight """
  intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
  union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
  jac = (intersection + smooth) / (union - intersection + smooth)
  return (1 - jac) * smooth

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.):
  y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
  epsilon = K.epsilon()
  y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
  loss = -alpha * K.pow(1. - y_pred, gamma) * y_true * K.log(y_pred)
  return K.mean(loss, axis=-1)

""" Focal Tversky Loss as seen in:
https://arxiv.org/pdf/1810.07842.pdf and adapted from https://github.com/nabsabraham/focal-tversky-unet/ """

def focal_tversky_loss(y_true, y_pred, alpha=0.3, inverted_gamma=0.5, smooth=1.):
  """
  alpha: weights FN and FP. Higher alpha weights FN higher. alpha=0.5 becomes dice score coefficient.
		     Paper uses 0.7 as the default to focus more on FN.

  inverted_gamma: focal parameter that is in the range 0.33 - 1 (i.e gamma is from 1 - 3).
				          Higher value makes the loss function focus more on easy examples. Lower value
				          makes the loss function focus more on less accurate predictions/misclassifications.
				   			  inverted_gamma=1 becomes tversky loss. Paper uses inverted_gamma=0.75.

  smooth: smoothing factor, usually set to 1.
  """
  def get_tversky_index(y_true, y_pred):
    TP = K.sum(y_true * y_pred)
    FN = K.sum(y_true * (1 - y_pred))
    FP = K.sum((1 - y_true) * y_pred)
    return (TP + smooth)/(TP + alpha*FN + (1 - alpha)*FP + smooth)

  tversky_index = get_tversky_index(y_true, y_pred)
  return K.pow(1 - tversky_index, inverted_gamma)

def combined_loss(y_true, y_pred, focal_loss_weight=0.25):
  return focal_loss_weight*focal_loss(y_true, y_pred) + (1. - focal_loss_weight)*focal_tversky_loss(y_true, y_pred)

