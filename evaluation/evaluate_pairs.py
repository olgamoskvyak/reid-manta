#Code is adapted from https://github.com/davidsandberg/facenet
import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
from sklearn import metrics
from scipy import interpolate
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

def evaluate_pairs(images, labels, model, far_target, plot_file, sample_size=None):
    """Evaluate model on pairs generated from  a set of images.
    Input:
    images: 4D numpy array, dtype=uint8
    labels: 1D numpy array of integer labels for images
    far_target: 
    """
    print('Test set: {} images, {} unique classes'.format(images.shape[0], np.unique(labels).shape[0]))
    #Get pairs and compute distances
    dist, actual_issame = model.compute_dist(images, labels, sample_size = sample_size)
    val, far, auc = evaluate_dist(dist, actual_issame, far_target, plot_file)
    print('VAL is {:.2f} when FAR is {:.3f}'.format(val, far))
    print('VAL - validation rate: ratio of true positive among all positive')
    print('FAR - false acceptance rate: ratio of false positives among all negative')
  
    return val, far, auc 
    

    
def evaluate_dist(distances, actual_issame, far_target, plot_file=None):
    """Evaluate on pairs if distances and labels are given
    Input:
    distances: array of floats (0,1); distance between pair.
    actual_issame: boolean, True for positive pairs, False for negative
    """
    fprs, tprs, ths = metrics.roc_curve(np.logical_not(actual_issame), distances)
    plot_roc(tprs, fprs, showFig = False, saveFig = True, figName=plot_file)  
    auc = metrics.auc(fprs, tprs)
    print('Area Under Curve (AUC): %1.3f' % auc)
    
    val, far = calculate_val_far_target(ths, distances, actual_issame, far_target)
    return val, far, auc


def calculate_val_far_target(thresholds, distances, actual_issame, far_target):
    
    nrof_thresholds = len(thresholds)

    val = 0.
    far = 0.

    # Find the threshold that gives FAR = far_target
    far_train = np.zeros(nrof_thresholds)
    for i, threshold in enumerate(thresholds):
        _, far_train[i] = calculate_val_far(threshold, distances, actual_issame)
        
    if np.max(far_train)>=far_target:
        f = interpolate.interp1d(far_train, thresholds, kind='slinear')
        threshold = f(far_target)
    else:
        threshold = 0.0

    val, far = calculate_val_far(threshold, distances, actual_issame)
    print('Threshold is set to {:.2f}'.format(threshold))

    return val, far    
    
    
def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def plot_roc(tprs, fprs, showFig = True, saveFig = False, figName = 'roc_curve.png'):
    plt.figure()
    lw = 2
    roc_auc = metrics.auc(fprs, tprs)
    plt.plot(fprs, tprs, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    if showFig:
        plt.show()
    if saveFig:
        plt.savefig(figName)
    plt.close()