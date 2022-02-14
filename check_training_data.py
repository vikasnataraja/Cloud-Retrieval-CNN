import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')
import argparse

def visualize_data(fdir):
	"""Check data before training the model"""
	
	# read in the data
	radiance = np.load(os.path.join(fdir, 'inp_radiance.npy'), allow_pickle=True).item()
	cot_true = np.load(os.path.join(fdir, 'out_cot_3d.npy'), allow_pickle=True).item()
	cot_ipa = np.load(os.path.join(fdir, 'out_cot_1d.npy'), allow_pickle=True).item()
	assert list(radiance.keys()) == list(cot_true.keys()), 'Image names of radiance and cot are different'

	key = 'data_10' # changeable
	
	# create figure of panels - radiance, ipa, true COT	
	fig = plt.figure(figsize=(25, 10))
	gs = GridSpec(1, 3, figure=fig)
    
	ax = fig.add_subplot(gs[0, 0])
	ax.tick_params(direction='out', length=10, width=2)
	ax.imshow(radiance[key], cmap='jet')
	ax.set_title('Radiance', fontsize=30)
	ax.set_xticks([0, 20, 40, 60])
	ax.set_xticklabels([0, 2, 4, 6], fontsize=24, fontweight="bold")
	ax.set_yticks([0, 20, 40, 60])
	ax.set_yticklabels([0, 2, 4, 6], fontsize=24, fontweight="bold")
	ax.set_xlabel('X [km]', fontsize=28, fontweight="bold")
	ax.set_ylabel('Y [km]', fontsize=28, fontweight="bold")
	
	ax = fig.add_subplot(gs[0, 1])
	ax.tick_params(direction='out', length=10, width=2)
	ax.imshow(cot_ipa[key], cmap='jet')
	ax.set_title('IPA COT', fontsize=30)
	ax.set_xticks([0, 20, 40, 60])
	ax.set_xticklabels([0, 2, 4, 6], fontsize=24, fontweight="bold")
	ax.set_yticks([])
	# ax.set_yticklabels([0, 2, 4, 6], fontsize=24, fontweight="bold")
	ax.set_xlabel('X [km]', fontsize=28, fontweight="bold")
	# ax.set_ylabel('Y [km]', fontsize=28, fontweight="bold")

	ax = fig.add_subplot(gs[0, 2])
	ax.tick_params(direction='out', length=10, width=2)
	ax.imshow(cot_true[key], cmap='jet')
	ax.set_title('True COT (Binned)', fontsize=30)
	ax.set_xticks([0, 20, 40, 60])
	ax.set_xticklabels([0, 2, 4, 6], fontsize=24, fontweight="bold")
	ax.set_yticks([])
	# ax.set_yticklabels([0, 2, 4, 6], fontsize=24, fontweight="bold")
	ax.set_xlabel('X [km]', fontsize=28, fontweight="bold")
	# ax.set_ylabel('Y [km]', fontsize=28, fontweight="bold")
	
	fig.savefig('sample_training_data_{}.png'.format(today), dpi=100, bbox_inches = 'tight', pad_inches = 0.25)
	print('Saved image as sample_training_data_{}.png'.format(today))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--fdir', default='data/npy/rgb_radiance', type=str,
                        help="Path to directory containing npy files")
	args = parser.parse_args()
	visualize_data(args.fdir)
 
