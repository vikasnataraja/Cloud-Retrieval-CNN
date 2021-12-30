import tensorflow as tf
from keras.layers import BatchNormalization, Layer


def BatchNorm():
    return BatchNormalization(momentum=0.95, epsilon=1e-5)


class UpSample(Layer):
    """ Custom Keras layer that upsamples to a new size using interpolation.
    Bypasses the use of Keras Lambda layer
    Args:
      - new_size: (tuple) new size to which layer needs to be resized to. Must be (height, width)
      - method: (str) method of interpolation to be used. If None, defaults to bilinear.
               Choose amongst 'bilinear', 'nearest', 'lanczos3', 'lanczos5', 'area', 'gaussian', 'mitchellcubic'
    Returns:
      - keras.layers.Layer of size [None, new_size[0], new_size[1], depth]
    """

    def __init__(self, new_size, method='bilinear', **kwargs):
        self.new_size = new_size
        self.method = method
        super(UpSample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(UpSample, self).build(input_shape)

    def call(self, inputs, **kwargs):
        resized_height, resized_width = self.new_size
        return tf.image.resize(images=inputs,
                               size=[resized_height, resized_width],
                               method=self.method,
                               align_corners=True)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(UpSample, self).get_config()
        config['new_size'] = self.new_size
        return config
