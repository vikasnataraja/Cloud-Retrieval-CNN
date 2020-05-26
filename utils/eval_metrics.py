import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

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

def plot_evaluation(target, prediction):
  """ plot the evaluation metrics """
  target_vals = np.unique(target)
  pred_vals = np.unique(prediction)
  precision, recall, f1_score, support = precision_recall_dice_support(target, prediction)

  fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,14))
  ax[0,0].plot(target_vals, precision*100, c='teal')
  ax[0,0].set_xlabel('Classes')
  ax[0,0].set_ylabel('Score')
  ax[0,0].set_xticks(target_vals)
  ax[0,0].set_yticks(np.linspace(0,100,5))
  ax[0,0].set_title('Precision per class')

  ax[0,1].plot(target_vals, recall*100, c='darkorange')
  ax[0,1].set_xlabel('Classes')
  ax[0,1].set_ylabel('Score')
  ax[0,1].set_xticks(target_vals)
  ax[0,1].set_yticks(np.linspace(0,100,5))
  ax[0,1].set_title('Recall per class')

  ax[1,0].plot(target_vals, f1_score*100, c='purple')
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
  plt.savefig('evaluation.png', dpi=100)
  plt.close();


