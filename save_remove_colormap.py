import glob
import os
import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--original_gt_dir', default=None, type=str, 
                    help="Path to original groundtruth images directory")
parser.add_argument('--output_dir', default=None, type=str, 
                    help="Path to output images directory where images will be saved")
parser.add_argument('--segmentation_format', default='png', type=str, 
                    help="Extension of the images to be saved")


args = parser.parse_args()

def _remove_colormap(filename):
  """Removes the color map from the annotation.
  Args:
    filename: Ground truth annotation filename.
  Returns:
    Annotation without color map.
  """
  return np.array(Image.open(os.path.join(args.original_gt_dir,filename)))


def _save_annotation(annotation, filename):
  """Saves the annotation as png file.
  Args:
    annotation: Segmentation annotation.
    filename: Output filename.
  """
  pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
  #with tf.io.gfile.GFile(filename, mode='w') as f:
  pil_image.save(os.path.join(args.output_dir,filename),args.segmentation_format)


def main():
  # Create the output directory if not exists.
  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  annotations = os.listdir(args.original_gt_dir)
  
  for annotation in annotations:
    raw_annotation = _remove_colormap(annotation)
    filename = os.path.splitext(annotation)[0]
    _save_annotation(raw_annotation,filename)
  print('Finished saving annotations in {} directory'.format(args.output_dir))


if __name__ == '__main__':
  main()