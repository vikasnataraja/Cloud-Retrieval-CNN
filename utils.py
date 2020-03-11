import numpy as np
import tensorflow as tf

def normalize(img):
	return (img-img.min())/(img.max()-img.min())
