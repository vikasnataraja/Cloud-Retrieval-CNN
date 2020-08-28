import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import tensorflow as tf
from utils.losses import focal_loss
from predict import predict_on_validation_set
import argparse


def ret_1d(input_data, cot_1d, cot_3d, validation_list=None):

    if validation_list is None:
        validation_list = list(input_data.keys())
    slopes1d, devs1d, means1d = [], [], []
    for i in validation_list:
        rad = input_data[i]
        numerator = cot_1d[i]
        gt_img = cot_3d[i]
        flat_pred = numerator.flatten()
        flat_gt = gt_img.flatten()

        non_zero_idx = np.where(flat_gt>0)[0] # indices that have non-zero classes
        non_zero_gt = flat_gt[non_zero_idx]
        non_zero_prediction = flat_pred[non_zero_idx]
        if non_zero_prediction.shape[0]==0:
            continue
        slope = non_zero_prediction/non_zero_gt
        slopes1d.append(np.mean(slope))
        devs1d.append(np.std(rad))
        means1d.append(np.mean(rad))

    return means1d, devs1d, slopes1d

def plot_1d_3d(means, stds, slopes1d, slopes3d, figsize=(16,6), label1='label_1', label2='label_2'):
    """ plot statistical metrics for 3D and 1D retrievals"""

    rows = 1
    cols = 3
    num_samples = len(means)

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(rows, cols,1)
    plt.scatter(means, stds, c='gold')
    plt.xlabel('RadMean')
    plt.ylabel('RadStd')
    plt.title('RadStd vs RadMean over {} samples'.format(num_samples))

    fig.add_subplot(rows, cols, 2)
    plt.scatter(stds, slopes1d, c='teal', label=label1, alpha=0.5)
    plt.scatter(stds, slopes3d, c='salmon', label=label2, alpha=0.5)
    plt.xlabel('RadStd')
    plt.ylabel('Slope')
    plt.title('RadStd vs Slope over {} samples'.format(num_samples))
    plt.grid()
    plt.legend()

    fig.add_subplot(rows, cols, 3)
    plt.scatter(means, slopes1d, c='teal', label=label1, alpha=0.5)
    plt.scatter(means, slopes3d, c='salmon', label=label2, alpha=0.5)
    plt.xlabel('RadMean')
    plt.ylabel('Slope')
    plt.title('RadMean vs Slope over {} samples'.format(num_samples))
    plt.legend()
    plt.grid()
    plt.show();


def main(input_file, file_1d, file_3d, modelpath):
    radiances = np.load('{}'.format(input_file), allow_pickle=True).item()
    cot_1d = np.load('{}'.format(file_1d), allow_pickle=True).item()
    cot_3d = np.load('{}'.format(file_3d), allow_pickle=True).item()
    
    model = load_model('{}'.format(modelpath), custom_objects={"tf":tf, "focal_loss":focal_loss})
    means3d, devs3d, slopes3d = predict_on_validation_set(radiances, cot_3d, model)
    print('\nThe mean slope of 3D retrievals is {}\n'.format(np.mean(slopes3d)))
    means1d, devs1d, slopes1d = ret_1d(radiances, cot_1d=cot_1d, cot_3d=cot_3d)
    print('The mean slope of 1D retrievals is {}\n'.format(np.mean(slopes1d)))
    plot_1d_3d(means1d, devs1d, slopes1d, slopes3d, label1='1d_ret', label2='3d_ret')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='data/single_channel/input_radiance.npy', type=str,
                        help="Path to numpy input images file")
    parser.add_argument('--file_1d', default=None, type=str, help="Path to the 1D retrievals numpy file")
    parser.add_argument('--file_3d', default=None, type=str, help="Path to the 3D retrievals numpy file")
    parser.add_argument('--model_path', default=None, type=str, help="Path to the model")
    args = parser.parse_args()

    main(args.input_file, args.file_1d, args.file_3d, args.modelpath)




