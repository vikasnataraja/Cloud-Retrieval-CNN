import numpy as np
from PIL import Image
import tensorflow as tf

def generate_color_map(N=256, normalized=False):
    """from https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae ."""
  def bitget(byteval, idx):
      return ((byteval & (1 << idx)) != 0)

  dtype = 'float32' if normalized else 'uint8'
  cmap = np.zeros((N, 3), dtype=dtype)
  for i in range(N):
      r = g = b = 0
      c = i
      for j in range(8):
          r = r | (bitget(c, 0) << 7 - j)
          g = g | (bitget(c, 1) << 7 - j)
          b = b | (bitget(c, 2) << 7 - j)
          c = c >> 3

      cmap[i] = np.array([r, g, b])

  cmap = cmap / 255 if normalized else cmap
  return cmap

def remove_groundtruth_colormap(filename):
  return np.array(Image.open(filename))

def save_with_colormap(filename):
  target = np.array(Image.open(filename))[:, :, np.newaxis]
  cmap = generate_color_map()[:, np.newaxis, :]
  new_im = np.dot(target == 0, cmap[0])
  for i in range(1, cmap.shape[0]):
    new_im += np.dot(target == i, cmap[i])
  new_im = Image.fromarray(new_im.astype(np.uint8))
  #blend_image = Image.blend(image, new_im, alpha=0.8)
  new_image.save('~/Desktop/tmp.jpg')