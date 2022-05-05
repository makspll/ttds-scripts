

from typing import Dict, List, Tuple
import numpy

import sklearn.metrics as metrics


def confusion_matrix(matrix : List[List]):
    """ accepts class labels in order of ascending column and row indices, 
    as well as a square confusion matrix of size len(class_labels)"""

    matrix = numpy.array(matrix)
    return matrix

def get_fake_sklearn_true_and_pred(confusion_matrix) -> Tuple:
    # we do a little hack, we convert a confusion matrix to fake list of predictions vs ground truths
    # this allows us to use all of sklearn's metrics

    y_pred = []
    y_true = []
    for i in range(len(confusion_matrix)): 
        for j in range(len(confusion_matrix)):
            y_pred.extend([j] * confusion_matrix[i,j])
            y_true.extend([i] * confusion_matrix[i,j])
    
    
    return y_true,y_pred

