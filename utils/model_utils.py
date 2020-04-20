import tensorflow as tf
import keras
from keras.layers import BatchNormalization

def BatchNorm():
  return BatchNormalization(momentum=0.95, epsilon=1e-5)

class UpSample(keras.layers.Layer):
  """ Custom Keras layer that upsamples to a new size using bilinear interpolation.
  Bypasses the use of Keras Lambda layer"""

  def __init__(self, new_size, **kwargs):
    self.new_size = new_size
    super(UpSample, self).__init__(**kwargs)

  def build(self, input_shape):
    super(UpSample, self).build(input_shape)

  def call(self, inputs, **kwargs):
    resized_height, resized_width = self.new_size
    return tf.image.resize(images=inputs,
                           size=[resized_height,resized_width],
                           method='bilinear',
                           align_corners=True)

  def compute_output_shape(self, input_shape):
    return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

  def get_config(self):
    config = super(UpSample, self).get_config()
    config['new_size'] = self.new_size
    return config
