#copied from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

import numpy as np
import keras.backend as K
import math

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(1, k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)], dtype=np.float32)

def acck(actual, prediction, k=5, verbose=True):
    """Computes accuracy at k
    Input:
    actual: list or 1D numpy array of true labels
    prediction: list of lists or 2D numpy array of predictions
    """
    accuracy_count = 0
    for i, true_lbl in enumerate(actual):
        if true_lbl in prediction[i][:k]:
            accuracy_count += 1
    accuracy = accuracy_count / len(actual)  
    if verbose:
        print('Predicting {} labels per input. Correct {} out of {}. ACC@{} %{:.2f}'.format(k, accuracy_count, len(actual), k, accuracy*100))
    return accuracy

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist

def contrastive_loss(y_true, y_pred, margin=1.0):
    """Contrastive loss for the Siamese architecture.
    """
    return K.mean((1.0 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))