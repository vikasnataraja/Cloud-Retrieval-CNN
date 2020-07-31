import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import os


def iou(target, prediction):
  """ calculate the iou between ground truth (target) and the predicted image """
  intersection = np.logical_and(target, prediction)
  union = np.logical_or(target, prediction)
  iou_score = np.sum(intersection) / np.sum(union)
  return iou_score


def precision_recall_dice_support(target, prediction):
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
  precision, recall, f1_score, support = precision_recall_dice_support(target, prediction)

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


def plot_1d_3d(means, stds, slopes1d, slopes3d, figsize-(16,6):
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
  plt.scatter(stds, slopes1d, c='teal', label='1d_ret',alpha=0.5)
  plt.scatter(stds, slopes3d, c='salmon', label='3d_ret',alpha=0.5)
  plt.xlabel('RadStd')
  plt.ylabel('Slope')
  plt.title('RadStd vs Slope over {} samples'.format(num_samples))
  plt.legend()

  fig.add_subplot(rows, cols, 3)
  plt.scatter(means, slopes1d, c='teal', label='1d_ret',alpha=0.5)
  plt.scatter(means, slopes3d, c='salmon', label='3d_ret',alpha=0.5)
  plt.xlabel('RadMean')
  plt.ylabel('Slope')
  plt.title('RadMean vs Slope over {} samples'.format(num_samples))
  plt.legend()
  plt.show()

  if not os.path.isdir('results/'):
    os.makedirs('results/')
  fig.savefig('results/1d_3d_comparison.png', dpi=100)
  print('Saved figure in "results/" as "1d_3d_comparison.png"')
  plt.close();


def draw_heatmap(means, devs, slopes1d, slopes3d, binsize=(50,50)):
  """ draw heatmap for 1D vs 3D retrievals """

  num_samples = len(means)
        
  fig = plt.figure(figsize=(15,25))
  ax1 = fig.add_subplot(3,1,1)
  ax1.scatter(means, devs, c='gold')
  ax1.set_title('Mean vs STD')
  ax1.set_xlabel('Mean')
  ax1.set_ylabel('RadStd')

  ax2 = fig.add_subplot(3,1,2)
  ax2.scatter(devs, slopes1d, c='teal', label='1d_ret')
  ax2.scatter(devs, slopes3d, c='salmon', label='3d_ret')
  ax2.set_xlabel('RadStd')
  ax2.set_ylabel('Slope')
  ax2.set_title('STD vs Slope over {} samples'.format(num_samples))
  ax2.legend()

  # ax3 = fig.add_subplot(4,2,4)
  # ax3.scatter(means, slopes1d, c='teal', label='1d_ret')
  # ax3.scatter(means, slopes3d, c='salmon', label='3d_ret')
  # ax3.set_xlabel('RadMean')
  # ax3.set_ylabel('Slope')
  # ax3.set_title('Mean vs Slope over {} samples'.format(num_samples))
  # ax3.legend()

  ax4 = fig.add_subplot(3,2,5)
  _, _ , _ , z1 = ax4.hist2d(devs, slopes1d, bins=binsize, cmap='jet')
  # ax4.set_xlim([-0.02, 0.17])
  ax4.set_xlabel('RadStd')
  ax4.set_ylabel('Slope')
  ax4.set_title('RadStd vs Slope Heatmap for 1D')
  fig.colorbar(z1, ax=ax4)

  ax6 = fig.add_subplot(3,2,6)
  _, _ , _ , z3 = ax6.hist2d(devs, slopes3d, bins=binsize, cmap='jet')
  ax6.set_xlabel('RadStd')
  ax6.set_ylabel('Slope')
  ax6.set_title('RadStd vs Slope Heatmap for 3D')
  fig.colorbar(z3, ax=ax6)
  plt.show()

  if not os.path.isdir('results/'):
    os.makedirs('results/')
  fig.savefig('results/heatmap.png', dpi=100)
  print('Saved heatmap figure in "results/" as "heatmap.png"')
  plt.close();
