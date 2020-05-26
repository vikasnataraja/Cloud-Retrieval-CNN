from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_layer(model, img, layer_idx, output_dir=os.getcwd(), fname=None, colormap=None, colorbar=False):
  """
  Visalizes feature maps of a layer of a saved Keras model and saved figure.

  Args:
	- model: Keras model.
	- img: arr, image for which feature maps needs to be visualized using the model.
	       Must be pre-processed and either 2D or 3D. Will be turned into a 4D tensor 
	       within this function.
	- layer_idx: int, index of the layer for which the feature maps need to be visualized.
	- output_dir: str, directory where the figure will be saved. If it doesn't exist,
		      it will be created
	- fname: str, filename of the saved figure. If None, will be saved under layer_idx name, 
		 e.g: layer_50.png
	- colormap: str, the colormap that has to be used during visualization. Has to be
		    one among matplotlib.pyplot's supported colormaps. If None, will default to
		    matploltlib's default 'viridis'.
	- colorbar: bool, set to True if colorbars need to be added beside each feature map.
		    Disabled by default.
  """

  if len(img.shape)>2:
    img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
  else:
    img = np.reshape(img,(1,img.shape[0],img.shape[1],1))
    
  model = Model(model.inputs, model.layers[layer_idx].output)
  activation_map = model.predict(img)

  # set columns to 6 to improve view
  cols = 6
  if activation_map.shape[-1] >= 64:
      rows = 6
  else:
      rows = int(activation_map.shape[-1]/cols)
  idx = 1
  fig = plt.figure(figsize=(16,12))
   
  for _ in range(rows):
    for _ in range(cols):
      # specify subplot and turn of axis
      ax = fig.add_subplot(rows, cols, idx)
      ax.set_xticks([])
      ax.set_yticks([])
      if colormap is not None:
        plt.imshow(activation_map[0, :, :, idx-1],cmap=colormap)
      else:
        plt.imshow(activation_map[0, :, :, idx-1])
      plt.title('Layer {} Filter {}'.format(layer_idx, idx-1))
      if colorbar:
        plt.colorbar()
      idx += 1
  plt.show()
  
  # create directory if it doesn't exist
  if not os.path.isdir(output_dir):
    print('Output directory {} does not exist,'\
          ' creating it now ...'.format(output_dir))
    os.makedirs(output_dir)

  # if no filename is given, use layer information
  if fname is None:
    fname = 'layer_{}'.format(layer_idx)

  # save figure
  fig.savefig('{}.png'.format(os.path.join(output_dir,fname)), dpi=100)
  print('Saved figure as {}.png'.format(os.path.join(output_dir,fname)))
  plt.close();
