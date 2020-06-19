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

def tversky_loss(y_true, y_pred, beta=0.5):
  numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
  return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)

"""
Weighted Cross Entropy
"""

def weighted_cross_entropy(y_true, y_pred):
  def convert_to_logits(y_pred):
    #see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    return tf.log(y_pred / (1 - y_pred))

  def wce_loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=35.0)
    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)
  return wce_loss(y_true, y_pred)

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

def focal_tversky(y_true, y_pred, alpha=0.7, gamma=0.75, smooth=1.):
  
  def tversky(y_true, y_pred):
    TP = K.sum(y_true * y_pred)
    FN = K.sum(y_true * (1-y_pred))
    FP = K.sum((1-y_true)*y_pred)
    return (TP + smooth)/(TP + alpha*FN + (1-alpha)*FP + smooth)

  tversky_index = tversky(y_true, y_pred)
  return K.pow(1-tversky_index, gamma)


