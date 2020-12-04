import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import h5py
import os
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import numpy.polynomial.polynomial as poly


def iou(target, prediction):
  """ calculate the iou between ground truth (target) and the predicted image """
  intersection = np.logical_and(target, prediction)
  union = np.logical_or(target, prediction)
  iou_score = np.sum(intersection) / np.sum(union)
  return iou_score

def correlation_coef(target, prediction):
  """ COT space correlation coefficient """
  return np.corrcoef(target.ravel(), prediction.ravel())[0,1]

def linear_reg_coeffs(target, prediction, deg=1):
  """ COT space linear regression coefficients -> A+Bx """
  coeffs = poly.polyfit(target.ravel(), prediction.ravel(), deg)
  return coeffs

def intersection_over_union(target, prediction):
  """ Class space IOU """
  return iou(target, prediction)

def slope(target, prediction):
  """ COT space slope """

  pred_flt = prediction.ravel()
  target_flt = target.ravel()

  non_zero_idx = np.where(target_flt > 0)[0] # indices that have non-zero classes
  non_zero_target = target_flt[non_zero_idx]
  non_zero_prediction = pred_flt[non_zero_idx]
  if non_zero_prediction.shape[0] != 0: # if all classes are zero, don't add to list
    return np.mean(non_zero_prediction/non_zero_target) # slope is element-wise division of non-zero classes


def get_precision_recall_dice_support(target, prediction):
  """ calculate precision, recall, f1_score and support """
  return precision_recall_fscore_support(target.flatten(), prediction.flatten(),
					 average=None, zero_division=0)


def visualize_prediction(input_img, target, prediction):
  """ visualize the input img, ground truth, predicted img and the diff"""

  rows = 1
  cols = 4

  fig = plt.figure(figsize=(16,16))
  fig.add_subplot(rows, cols,1)
  plt.imshow(input_img, cmap='plasma')
  plt.title('Radiance channel')

  fig.add_subplot(rows, cols, 2)
  plt.imshow(target, cmap='plasma')
  # plt.title('Ground truth COT = {}'.format(np.unique(target)[0]))
  plt.title('Ground truth COT')

  fig.add_subplot(rows, cols, 3)
  plt.imshow(prediction, cmap='plasma')
  # plt.title('Predicted COT = {}'.format(np.unique(prediction)[0]))
  plt.title('Predicted COT')

  fig.add_subplot(rows, cols, 4)
  plt.imshow(np.abs(prediction-target), cmap='plasma')
  plt.title('Diff Map')
  plt.show()
  if not os.path.isdir('results/'):
    os.makedirs('results/')
  fig.savefig('results/visualization.png', dpi=100)
  print('Saved visualized figure in "results/" as "visualization.png"')
  plt.close();


def plot_evaluation(target, prediction):
  """ plot the evaluation metrics """
  target_vals = np.unique(target)
  pred_vals = np.unique(prediction)
  precision, recall, f1_score, support = get_precision_recall_dice_support(target, prediction)

  fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,14))
  ax[0,0].plot(precision*100, c='teal')
  ax[0,0].set_xlabel('Classes')
  ax[0,0].set_ylabel('Score')
  ax[0,0].set_xticks(target_vals)
  ax[0,0].set_yticks(np.linspace(0,100,5))
  ax[0,0].set_title('Precision per class')

  ax[0,1].plot(recall*100, c='darkorange')
  ax[0,1].set_xlabel('Classes')
  ax[0,1].set_ylabel('Score')
  ax[0,1].set_xticks(target_vals)
  ax[0,1].set_yticks(np.linspace(0,100,5))
  ax[0,1].set_title('Recall per class')

  ax[1,0].plot(f1_score*100, c='purple')
  ax[1,0].set_xlabel('Classes')
  ax[1,0].set_ylabel('Score')
  ax[1,0].set_xticks(target_vals)
  ax[1,0].set_yticks(np.linspace(0,100,5))
  ax[1,0].set_title('F1 Score per class')

  ax[1,1].scatter(target.flatten(), prediction.flatten(), c='navy')
  stop = target_vals.max()
  ax[1,1].plot(np.linspace(0,stop), np.linspace(0,stop), c='black')
  ax[1,1].set_xlabel('Ground Truth Classes')
  ax[1,1].set_ylabel('Predicted Classes')
  ax[1,1].set_xticks(target_vals)
  ax[1,1].set_yticks(pred_vals)
  ax[1,1].set_title('Intersection over Union (IoU): {:0.2f}%'.format(iou(target, prediction)*100))

  plt.show()
  if not os.path.isdir('results/'):
    os.makedirs('results/')
  fig.savefig('results/evaluation.png', dpi=100)
  print('Saved evaluation figure in "results/" as "evaluation.png"')
  plt.close();


def plot_stat_metrics(means, stds, slopes):
  """ plot statistical metrics for a given set"""

  rows = 1
  cols = 3
  num_samples = len(means)

  fig = plt.figure(figsize=(16,6))
  fig.add_subplot(rows, cols,1)
  plt.scatter(means, stds, c='teal')
  plt.xlabel('RadMean')
  plt.ylabel('RadStd')
  plt.title('STD vs Mean dependence over {} samples'.format(num_samples))

  fig.add_subplot(rows, cols, 2)
  plt.scatter(stds, slopes, c='teal')
  plt.xlabel('RadStd')
  plt.ylabel('Slope')
  plt.title('STD vs Slope over {} samples'.format(num_samples))

  fig.add_subplot(rows, cols, 3)
  plt.scatter(means, slopes, c='teal')
  plt.xlabel('RadMean')
  plt.ylabel('Slope')
  plt.title('Mean vs Slope over {} samples'.format(num_samples))

  plt.show()
  if not os.path.isdir('results/'):
    os.makedirs('results/')
  fig.savefig('results/stat_evaluation.png', dpi=100)
  print('Saved stat evaluation figure in "results/" as "stat_evaluation.png"')
  plt.close();


def plot_model_comparison(means, stds, slopes_1, slopes_2, figname, figsize=(16,6),
               label1='label_1', label2='label_2', transparency=0.5, markersize=1.8):
  """ plot statistical metrics for two model retrievals"""

  rows = 1
  cols = 2
  num_samples = len(means)

  fig = plt.figure(figsize=figsize)
  fig.add_subplot(rows, cols, 1)
  plt.scatter(means, stds, c='gold', s=markersize)
  plt.xlabel('RadMean')
  plt.ylabel('RadStd')
  plt.title('RadMean vs RadStd over {} samples'.format(num_samples))

  # fig.add_subplot(rows, cols, 2)
  # plt.scatter(stds, slopes_1, c='teal', label=label1, alpha=transparency, s=markersize)
  # plt.scatter(stds, slopes_2, c='salmon', label=label2, alpha=transparency, s=markersize)
  # plt.xlabel('RadStd')
  # plt.ylabel('Slope')
  # plt.title('RadStd vs Slope over {} samples'.format(num_samples))
  # plt.grid()
  # plt.legend()

  fig.add_subplot(rows, cols, 2)
  plt.scatter(means, slopes_1, c='teal', label=label1, alpha=transparency, s=markersize)
  plt.scatter(means, slopes_2, c='salmon', label=label2, alpha=transparency, s=markersize)
  plt.xlabel('RadMean')
  plt.ylabel('Slope')
  plt.title('RadMean vs Slope over {} samples'.format(num_samples))
  plt.grid()
  plt.legend()
  plt.show()

  if not os.path.isdir('results/'):
    os.makedirs('results/')
  fig.savefig('results/{}'.format(figname), dpi=100)
  print('Saved figure in "results/" as "{}"'.format(figname))


def plot_all(rad_cot_space, cot_true_cot_space, cot_true_class_space,
             prediction_cot_space, prediction_class_space,
             cot_1d_cot_space, cot_1d_class_space,
             rows, filename, random=False, hist_bins=None, dimensions='64x64',
             figsize=(42,30)):

  cols = 5
  if hist_bins is None:
    hist_bins = np.linspace(0.0, 100.0, 100)

  fig = plt.figure(figsize=figsize)
  spec = fig.add_gridspec(nrows=rows, ncols=cols)

  for i in range(rows):
    key = i
    if random:
        key = np.random.choice(list(prediction_cot_space.keys()))

    input_img = rad_cot_space[key]
    truth = cot_true_cot_space[key]
    truth_class = cot_true_class_space[key]
    pred_cnn = prediction_cot_space[key] #cot space
    pred_cnn_class = prediction_class_space[key] #class space

    pred_1d = cot_1d_cot_space[key] #cot space
    pred_1d_class = cot_1d_class_space[key] #class space

    normalize = matplotlib.colors.Normalize(vmin=0, vmax=max(truth.max(), pred_cnn.max(), pred_1d.max()))

    ax0 = fig.add_subplot(spec[i, 0])
    x = ax0.imshow(input_img, cmap='jet')
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title('Radiance ({})'.format(dimensions), fontsize=15)

    inner = gridspec.GridSpecFromSubplotSpec(ncols=1, nrows=3, subplot_spec=spec[i,1], wspace=0.2, hspace=0.2)

    ax10 = fig.add_subplot(inner[0])
    y = ax10.imshow(truth, cmap='jet', norm=normalize)
    ax10.set_title('Gnd. Truth')
    ax10.set_xticks([])
    ax10.set_yticks([])
    divider = make_axes_locatable(ax10)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(y, cax=cax, ax=ax10)

    ax11 = fig.add_subplot(inner[1])
    z = ax11.imshow(pred_1d, cmap='jet', norm=normalize)
    ax11.set_title('1D COT')
    ax11.set_xticks([])
    ax11.set_yticks([])
    divider = make_axes_locatable(ax11)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(z, cax=cax, ax=ax11)

    ax12 = fig.add_subplot(inner[2])
    z = ax12.imshow(pred_cnn, cmap='jet', norm=normalize)
    ax12.set_title('Predicted COT')
    ax12.set_xticks([])
    ax12.set_yticks([])
    divider = make_axes_locatable(ax12)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(z, cax=cax, ax=ax12)

    ax3 = fig.add_subplot(spec[i, 2])
    ax3.scatter(truth.ravel(), pred_cnn.ravel(), c='red', s=30, lw=0.0, alpha=0.8)
    ax3.scatter(truth.ravel(), pred_1d.ravel(), c='green', s=30, lw=0.0, alpha=0.7)
    ax3.plot([0, truth.max()], [0, max(pred_cnn.max(),pred_1d.max())], c='black', ls='--')
    ax3.set_xlabel('cot ground truth')
    ax3.set_ylabel('cot')
    ax3.set_title('cot vs. ground truth', fontsize=15)
    patches_legend = [
                matplotlib.patches.Patch(color='red' , label='pred'),
                matplotlib.patches.Patch(color='green' , label='1d retrv.'),
                ]
    ax3.legend(handles=patches_legend, loc='upper left', fontsize=12)

    ax4 = fig.add_subplot(spec[i, 3])
    ax4.hist(truth.ravel(), bins=hist_bins, color='black', lw=0.0, alpha=0.5, density=True, histtype='stepfilled')
    ax4.hist(pred_1d.ravel() , bins=hist_bins, color='green', lw=2.0, alpha=0.8, density=True, histtype='step')
    ax4.hist(pred_cnn.ravel(), bins=hist_bins, color='red' , lw=1.0, alpha=0.8, density=True, histtype='step')
    background_pct = pred_1d[pred_1d<1].shape[0]*100/pred_1d.size
    if background_pct < 70:
      ax4.set_xlim((0, 100))
    else:
      ax4.set_xlim((0, 20))
    #     ax[i,3].set_ylim((0.001, 1.0))
    ax4.set_xlabel('COT')
    ax4.set_ylabel('Linear frequency')
    patches_legend = [
                matplotlib.patches.Patch(color='black' , label='Gnd. truth'),
                matplotlib.patches.Patch(color='green' , label='1D retrv.'),
                matplotlib.patches.Patch(color='red'   , label='Pred'),
                    ]
    ax4.legend(handles=patches_legend, loc='upper right', fontsize=12)

    ax5 = fig.add_subplot(spec[i, 4])
    ax5.hist(truth.ravel(), bins=hist_bins, color='black', lw=0.0, alpha=0.5, density=True, histtype='stepfilled')
    ax5.hist(pred_1d.ravel() , bins=hist_bins, color='green', lw=2.0, alpha=0.8, density=True, histtype='step')
    ax5.hist(pred_cnn.ravel(), bins=hist_bins, color='red' , lw=1.0, alpha=0.8, density=True, histtype='step')
    ax5.set_yscale('log')
    ax5.set_xlim((0, 100))
    ax5.set_ylim((0.001, 1.0))
    ax5.set_xlabel('COT')
    ax5.set_ylabel('Log frequency')
    ax5.set_title('CNN:(IoU: {:0.2f}, r: {:0.2f}, Slope: {:0.2f}, A: {:0.2f}, B: {:0.2f})\n'
                  '1D:(IoU: {:0.2f}, r: {:0.2f}, Slope: {:0.2f}, A: {:0.2f}), B: {:0.2f})'.format(intersection_over_union(truth_class, pred_cnn_class),
                                                                                 correlation_coef(truth, pred_cnn),
                                                                                 slope(truth, pred_cnn),
                                                                                 linear_reg_coeffs(truth, pred_cnn)[0], linear_reg_coeffs(truth, pred_cnn)[1],
                                                                                 intersection_over_union(truth_class, pred_1d_class),
                                                                                 correlation_coef(truth, pred_1d),
                                                                                 slope(truth, pred_1d),
                                                                                 linear_reg_coeffs(truth, pred_1d)[0], linear_reg_coeffs(truth, pred_1d)[1]))
    ax5.legend(handles=patches_legend, loc='upper right', fontsize=12)



  plt.subplots_adjust(wspace=0.15, hspace=0.3)
  if not os.path.isdir('results/'):
    os.makedirs('results/')
  fig.savefig('results/{}'.format(filename), dpi=100)
  print('Saved figure in "results/" as "{}"'.format(filename))
  plt.show()
  plt.close();


