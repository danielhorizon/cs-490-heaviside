import numpy as np 
import pandas as pd 

from sklearn.metrics import accuracy_score, precision_score, recall_score


def _accuracy(gt, pt):
    gt, pt = np.array(gt), np.array(pt)
    return np.mean(gt == pt)


def _accuracy_with_threshold(gt, pt, threshold): 
    ''' For continuous predictions. 
    '''
    gt = [0 if x >= threshold else 1 for x in gt]
    pt = [0 if x >= threshold else 1 for x in pt]
    return accuracy_score(gt=gt, pt=pt)


def _precision(gt, pt, class_value):
    """ Calculating precision = tp / (tp + fp)
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
    What proportion of predicted positives are truly positive? 
    Classifier's ability to not label as positive a sample that is negative.
    """
    tp = len([a for a, p in zip(gt, pt) if a == p and p == class_value])
    fp = len([a for a, p in zip(gt, pt) if a != p and p == class_value])
    # return tp / (tp + fp)
    return tp, fp


def _overall_precision(gt, pt, average=None):
    """precision = tp / (tp + fp)
    """
    total_tp, total_fp = 0, 0
    classes = list(set(gt))
    class_precisions = [] 
    weighted_precisions = [] 

    for c in classes:
        tp, fp = _precision(gt, pt, class_value=c) 
        total_tp += tp 
        total_fp += fp 
        
        # saving values 
        class_precision = tp / (tp + fp)
        class_precisions.append(class_precision)
        weighted_precision = class_precision * gt.count(c)
        weighted_precisions.append(weighted_precision)

    if average == "micro":
        return total_tp / (total_tp + total_fp)
    if average == "macro": 
        return np.mean(np.array(class_precisions))
    # average weighted by their support (# of true instances for each label)
    if average == "weighted": 
        return sum(weighted_precisions)/len(gt)


# Recall = TP / (TP+FN)
def _recall(gt, pt, class_value): 
    """recall = tp / (tp + fn)
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
    Classifier's ability to find all the positive samples 
    """
    tp = len([a for a, p in zip(gt, pt) if a == p and p == class_value])
    fn = len([a for a, p in zip(gt, pt) if a != p and a == class_value])
    return tp, fn

def _overall_recall(gt, pt, average=None):
    total_tp, total_fn = 0, 0
    classes = list(set(gt))
    class_recalls = [] 
    weighted_recalls = [] 

    for c in classes:
        tp, fn = _recall(gt, pt, class_value=c)
        total_tp += tp 
        total_fn += fn
        
        # saving values 
        class_recall = tp / (tp + fn)
        class_recalls.append(class_recall)
        weighted_recall = class_recall * gt.count(c)
        weighted_recalls.append(weighted_recall)

    if average == "micro":
        return total_tp / (total_tp + total_fn)
    if average == "macro": 
        return np.mean(np.array(class_recalls))
    # average weighted by their support (# of true instances for each label)
    if average == "weighted": 
        return sum(weighted_recalls)/len(gt)


def _f1_score(gt, pt, average=None): 
    '''
    F1 = 2 * (precision * recall) / (precision + recall)
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    In the multi-class and multi-label case, this is the average of
    the F1 score of each class with weighting depending on the ``average``
    parameter.
    '''
    


def test():
    gt = [0, 1, 2, 3, 0, 1, 2, 3, 4]
    pt = [0, 1, 3, 2, 0, 1, 3, 2, 4]
    # Accuracy
    print(_accuracy(gt, pt))       
    print(accuracy_score(gt, pt))

    y_true = [0, 1, 2, 0, 1 ,2, 3, 3]
    y_pred = [0, 2, 1, 0, 0, 1, 3, 1]
    print("PRECISION")
    print("daniel: {}, sklearn :{}".format(
        _overall_precision(y_true, y_pred, average="micro"), 
        precision_score(y_true, y_pred, average="micro")))
    print("daniel: {}, sklearn :{}".format(
        _overall_precision(y_true, y_pred, average="macro"),
        precision_score(y_true, y_pred, average="macro")))
    print("daniel: {}, sklearn :{}".format(
        _overall_precision(y_true, y_pred, average="weighted"),
        precision_score(y_true, y_pred, average="weighted")))


    print("\n")
    print("RECALL")
    print("daniel: {}, sklearn :{}".format(
        _overall_recall(y_true, y_pred, average="micro"),
        recall_score(y_true, y_pred, average="micro")))
    print("daniel: {}, sklearn :{}".format(
        _overall_recall(y_true, y_pred, average="macro"),
        recall_score(y_true, y_pred, average="macro")))
    print("daniel: {}, sklearn :{}".format(
        _overall_recall(y_true, y_pred, average="weighted"),
        recall_score(y_true, y_pred, average="weighted")))

if __name__ == '__main__':
    test()
