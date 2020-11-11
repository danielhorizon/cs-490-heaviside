import numpy as np 
import pandas as pd 


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
# https://docs.cloud.oracle.com/en-us/iaas/tools/ads-sdk/latest/user_guide/eval/Multiclass.html
# https://blog.floydhub.com/a-pirates-guide-to-accuracy-precision-recall-and-other-scores/#recall


def get_confusion_2(gt, pt, class_value=None): 
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
           TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
           FP += 1
        if y_actual[i] == y_hat[i] == 0:
           TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
           FN += 1
    return(TP, FP, TN, FN)


def get_confusion(gt, pt, class_value=None):
    """ Getting tp, fp, fn, tn for a class. 
    """
    tp = len([a for a, p in zip(gt, pt) if a == p and p == class_value])
    fp = len([a for a, p in zip(gt, pt) if a != p and p == class_value])
    fn = len([a for a, p in zip(gt, pt) if a != p and a == class_value])
    tn = len([a for a, p in zip(gt, pt) if a == p and p != class_value])
    return tp, fp, fn, tn 


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
    tp, fp, _, _ = get_confusion(gt, pt, class_value=class_value)
    return (tp / (tp + fp))


def _overall_precision(gt, pt, average=None):
    # for each row  in predictions 
    # row =  [1.0638e-03, 5.6466e-01, 4.3428e-01],
    # what is the gt for this  row? prob class b 
    # gt = [0, 1, 0]

    # comparing 0.00106 to  0

    # they're not independent for sigmoid,  if you take a sigmoid  of each one of it, 
    # you create a bunch of binary classifiers which aren't all output. 

    """precision = tp / (tp + fp)
    """
    total_tp, total_fp = 0, 0
    classes = list(set(gt))
    class_precisions = []
    weighted_precisions = []

    for c in classes:
        tp, fp, _, _ = get_confusion(gt, pt, class_value=c)
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
    tp, _, fn, _ = get_confusion(gt, pt, class_value=class_value)
    return (tp / (tp + fn))


def _overall_recall(gt, pt, average=None):
    total_tp, total_fn = 0, 0
    classes = list(set(gt))
    class_recalls = [] 
    weighted_recalls = [] 

    for c in classes:
        tp, _, fn, _ = get_confusion(gt, pt, class_value=c)
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
    classes = list(set(gt))
    f1_scores, wt_f1_scores = [], [] 

    for c in classes: 
        recall = _recall(gt, pt, class_value=c)
        precision = _precision(gt, pt, class_value=c)

        f1_score = 2 * (precision * recall) / (precision + recall)
        wt_f1_score = f1_score * gt.count(c)
        f1_scores.append(f1_score)
        wt_f1_scores.append(wt_f1_score)

    if average == "macro": 
        return np.mean(np.array(f1_scores))
    if average == "weighted": 
        return sum(wt_f1_scores)/len(gt)
    else: 
        # return macro as default
        return np.mean(np.array(f1_scores))


def _overall_f1_score(gt, pt, average=None): 
    '''
    F1 = 2 * (precision * recall) / (precision + recall)
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    In the multi-class and multi-label case, this is the average of
    the F1 score of each class with weighting depending on the ``average``
    parameter.
    '''
    if average == "micro": 
        micro_p = _overall_precision(gt, pt, average="micro")
        micro_r = _overall_recall(gt, pt, average="micro")
        return 2 * (micro_p * micro_r) / (micro_p + micro_r)

    # average of per-class f1 scores
    if average == "macro": 
        return _f1_score(gt, pt, average="macro")

    if average == "weighted": 
        return _f1_score(gt, pt, average="weighted")


def compute_confusion_matrix(true, pred):
    '''
    Results are identical (and similar in computation time) to: 
    "from sklearn.metrics import confusion_matrix"
    '''
    K = len(np.unique(true)) # Number of classes 
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
    return result


def create_conf_matrix(expected, predicted, n_classes):
    m = [[0] * n_classes for i in range(n_classes)]
    for pred, exp in zip(predicted, expected):
        m[pred][exp] += 1
    return m


def _kappa_score(gt, pt): 
    """
    po = relative observed agreement among raters 
    pe = hypothetical probability of chance agreement 
     https://stats.stackexchange.com/questions/251165/cohens-kappa-with-three-categories-of-variable
    """
    classes = list(set(gt))

    # computing using confusion matrix 
    cm = compute_confusion_matrix(true=gt, pred=pt)
    col_totals = [sum(x) for x in zip(*cm)]

    # computing num of agreements
    agreements = np.trace(cm)
    total_el = sum(map(sum, cm))
    ef_arr = [] 
    for i in range(len(classes)):
        row_sum = np.sum(cm[i]) # ith row
        col_sum = col_totals[i] # ith col
        ef = (row_sum*col_sum) / (total_el)
        ef_arr.append(ef)
    
    K = (agreements - np.sum(ef_arr)) / (total_el - np.sum(ef_arr))
    return K


def area(x, y):
    ''' area under curve via trapezoidal rule '''
    direction = 1
    # the following is equivalent to: dx = np.diff(x)
    dx = x[1:] - x[:-1]
    if torch.any(dx < 0):
        if torch.all(dx <= 0):
            direction = -1
        else:
            logging.warn(
                "x is neither increasing nor decreasing\nx: {}\ndx: {}.".format(x, dx))
            return 0
    return direction * torch.trapz(y, x)


def _auroc_score(gt, pt): 
    # TODO(dlee): finish this if needed. 
    """
    ROC - summarizes tradeoff between TPR and FPR 
    PR - Tradeoff between P and R for different thresholds
    """
    classes = list(set(gt))
    areas = []
    for c in classes:
        tp, fp, fn, tn = get_confusion(gt, pt, class_value=c)
        fpr = fp/(fp+tn+EPS)
        tpr = tp/(tp+fn+EPS)
        a = area(fpr, tpr)
        areas.append(a)
    return np.mean(np.array(areas))


# def mean_f1_approx_loss_on(threshold=None):




################################################################################
# def legacy_mean_f1_approx_loss_on(thresholds=torch.arange(0.1, 1, 0.1)):
#     def loss(pt, gt):
#         """Approximate F1:
#             - Linear interpolated Heaviside function 
#             - Harmonic mean of precision and recall
#             - Mean over a range of thresholds

#         We observe that H(p,tau) can be replaced with a reasonably-sized O(1) lookup table by 
#         truncating p to several decimal places and precomputing H for values of p and tau over the 
#         range [0, 1]. 
#         """
#         classes = pt.shape[1]
#         mean_f1s = torch.zeros(classes, dtype=torch.float32)
#         # mean over all classes
#         for i in range(classes):
#             thresholds = torch.arange(0.1, 1, 0.1)
#             # returns the number of tp, fn, fp, and tn.
#             tp, fn, fp, _ = confusion(
#                 gt, pt[:, i] if classes > 1 else pt, thresholds)
#             precision = tp/(tp+fp+EPS)
#             recall = tp/(tp+fn+EPS)
#             mean_f1s[i] = torch.mean(
#                 2 * (precision * recall) / (precision + recall + EPS))
#         loss = 1 - mean_f1s.mean()
#         return loss
#     return loss


# def legacy_mean_f1_approx_loss_on(thresholds=torch.arange(0.1, 1, 0.1)):
#     def loss(pt, gt):
#         """Approximate F1:
#             - Linear interpolated Heaviside function 
#             - Harmonic mean of precision and recall
#             - Mean over a range of thresholds

#         We observe that H(p,tau) can be replaced with a reasonably-sized O(1) lookup table by 
#         truncating p to several decimal places and precomputing H for values of p and tau over the 
#         range [0, 1]. 
#         """
#         classes = pt.shape[1]
#         mean_f1s = torch.zeros(classes, dtype=torch.float32)
#         # mean over all classes
#         for i in range(classes):
#             thresholds = torch.arange(0.1, 1, 0.1)
#             # returns the number of tp, fn, fp, and tn.
#             tp, fn, fp, _ = confusion(
#                 gt, pt[:, i] if classes > 1 else pt, thresholds)
#             precision = tp/(tp+fp+EPS)
#             recall = tp/(tp+fn+EPS)
#             mean_f1s[i] = torch.mean(
#                 2 * (precision * recall) / (precision + recall + EPS))
#         loss = 1 - mean_f1s.mean()
#         return loss
#     return loss


# def legacy_mean_accuracy_approx_loss_on(thresholds=torch.arange(0.1, 1, 0.1)):
#     def loss(pt, gt):
#         """Approximate Accuracy:
#             - Linear interpolated Heaviside function
#             - (TP + TN) / (TP + TN + FP + FN)
#             - Mean over a range of thresholds
#         """
#         classes = pt.shape[1]
#         mean_accs = torch.zeros(classes, dtype=torch.float32)
#         # mean over all classes
#         for i in range(classes):
#             tp, fn, fp, tn = confusion(
#                 gt, pt[:, i] if classes > 1 else pt, thresholds)
#             mean_accs[i] = torch.mean((tp + tn) / (tp + tn + fp + fn))
#         loss = 1 - mean_accs.mean()
#         return loss
#     return loss


# def area(x, y):
#     ''' area under curve via trapezoidal rule'''
#     direction = 1
#     # the following is equivalent to: dx = np.diff(x)
#     dx = x[1:] - x[:-1]
#     if torch.any(dx < 0):
#         if torch.all(dx <= 0):
#             direction = -1
#         else:
#             logging.warn(
#                 "x is neither increasing nor decreasing\nx: {}\ndx: {}.".format(x, dx))
#             return 0
#     return direction * torch.trapz(y, x)


# def legacy_mean_auroc_approx_loss_on(linspacing=11):
#     def loss(pt, gt):
#         """Approximate auroc:
#             - Linear interpolated Heaviside function
#             - roc (11-point approximation)
#             - integrate via trapezoidal rule under curve
#         """
#         classes = pt.shape[1]
#         thresholds = torch.linspace(0, 1, linspacing)
#         areas = []
#         # mean over all classes
#         for i in range(classes):
#             tp, fn, fp, tn = confusion(
#                 gt, pt[:, i] if classes > 1 else pt, thresholds)
#             fpr = fp/(fp+tn+EPS)
#             tpr = tp/(tp+fn+EPS)
#             a = area(fpr, tpr)
#             if a > 0:
#                 areas.append(a)
#         loss = 1 - torch.stack(areas).mean()
#         return loss
#     return loss


# compute metric value from cunfusion matrix
def compute_metric_from_cm(metric, C_val):
    # check for special cases
    if metric.special_case_positive:
        if C_val.ap == 0 and C_val.pp == 0:
            return 1.0
        elif C_val.ap == 0:
            return 0.0
        elif C_val.pp == 0:
            return 0.0

    if metric.special_case_negative:
        if C_val.an == 0 and C_val.pn == 0:
            return 1.0
        elif C_val.an == 0:
            return 0.0
        elif C_val.pn == 0:
            return 0.0

    val = metric.metric_expr.compute_value(C_val)
    return val




def test():
    gt = [0, 1, 2, 3, 0, 1, 2, 3, 4, 3, 2, 1]
    pt = [0, 1, 3, 2, 0, 1, 3, 2, 4, 3, 2, 2]
    # Accuracy
    print("--- ACCURACY") 
    print("daniel: {}, sklearn: {}".format(_accuracy(gt, pt), accuracy_score(gt, pt)))

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

    print("--- RECALL")
    print("daniel: {}, sklearn :{}".format(
        _overall_recall(y_true, y_pred, average="micro"),
        recall_score(y_true, y_pred, average="micro")))
    print("daniel: {}, sklearn :{}".format(
        _overall_recall(y_true, y_pred, average="macro"),
        recall_score(y_true, y_pred, average="macro")))
    print("daniel: {}, sklearn :{}".format(
        _overall_recall(y_true, y_pred, average="weighted"),
        recall_score(y_true, y_pred, average="weighted")))

    print("--- F1 Score")
    print("MICRO: daniel: {}, sklearn :{}".format(
        _overall_f1_score(gt, pt, average="micro"),
        f1_score(gt, pt, average="micro")))
    print("MICRO: daniel: {}, sklearn :{}".format(
        _overall_f1_score(y_true, y_pred, average="micro"),
        f1_score(y_true, y_pred, average="micro")))
    print("daniel: {}, sklearn :{}".format(
        _overall_f1_score(gt, pt, average="macro"),
        f1_score(gt, pt, average="macro")))
    print("daniel: {}, sklearn :{}".format(
        _overall_f1_score(gt, pt, average="weighted"),
        f1_score(gt, pt, average="weighted")))

    print("--- Kappa Score")
    print("daniel: {}, sklearn :{}".format(
        _kappa_score(gt, pt),
        cohen_kappa_score(gt, pt)))


if __name__ == '__main__':
    test()


