import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from heavisidefunctions import *

HEAVISIDE = ['f1_05_sig_k10', 'f1_05_sig_k20', 'f1_05_sig_fit'] + \
    ['f1_mean', 'f1_mean_sig_k10', 'f1_mean_sig_k20', 'f1_mean_sig_fit']
REPORTED = ['bce', 'roc', 'accuracy_05', 'f1_max', 'f1_05', 'f2_05', 'f3_05']
ALL = REPORTED + ['accuracy', 'auroc', 'f2_mean', 'f3_mean'] + HEAVISIDE

def get(loss):
    if loss == 'bce':
        return tf.keras.losses.BinaryCrossentropy()
    elif loss == 'accuracy':
        return accuracy_approx()
    elif loss == 'accuracy_05':
        return accuracy_at(thresholds=tf.constant([0.5]), heaviside=linear_heaviside)
    elif loss == 'roc':
        return roc_auc_score
    elif loss == 'auroc':
        return auroc_approx
    elif loss == 'auroc_mean':
        return auroc_approx_mean
    elif loss == 'f1_mean':
        return mean_fb_approx_at(b=1.0, thresholds=tf.range(0.1, 1, 0.1), heaviside=linear_heaviside)
    elif loss == 'f1_mean_sig_k10':
        return mean_fb_approx_at(b=1.0, thresholds=tf.range(0.1, 1, 0.1), heaviside=sigmoid_heaviside(k=10))
    elif loss == 'f1_mean_sig_k20':
        return mean_fb_approx_at(b=1.0, thresholds=tf.range(0.1, 1, 0.1), heaviside=sigmoid_heaviside(k=20))
    elif loss == 'f1_mean_sig_fit':
        return mean_fb_approx_at(b=1.0, thresholds=tf.range(0.1, 1, 0.1), heaviside=fit_sigmoid_heaviside)
    elif loss == 'f1_05':
        return mean_fb_approx_at(b=1.0, thresholds=tf.constant([0.5]), heaviside=linear_heaviside)
    elif loss == 'f1_05_sig_k10':
        return mean_fb_approx_at(b=1.0, thresholds=tf.constant([0.5]), heaviside=sigmoid_heaviside(k=10))
    elif loss == 'f1_05_sig_k20':
        return mean_fb_approx_at(b=1.0, thresholds=tf.constant([0.5]), heaviside=sigmoid_heaviside(k=20))
    elif loss == 'f1_05_sig_fit':
        return mean_fb_approx_at(b=1.0, thresholds=tf.constant([0.5]), heaviside=fit_sigmoid_heaviside)
    elif loss == 'f2_mean':
        return mean_fb_approx_at(b=2.0, thresholds=tf.range(0.1, 1, 0.1), heaviside=linear_heaviside)
    elif loss == 'f2_05':
        return mean_fb_approx_at(b=2.0, thresholds=tf.constant([0.5]), heaviside=linear_heaviside)
    elif loss == 'f3_mean':
        return mean_fb_approx_at(b=3.0, thresholds=tf.range(0.1, 1, 0.1), heaviside=linear_heaviside)
    elif loss == 'f3_05':
        return mean_fb_approx_at(b=3.0, thresholds=tf.constant([0.5]), heaviside=linear_heaviside)
    elif loss == 'f1_max':
        return max_fb_approx_at(b=1.0, thresholds=tf.range(0.1, 1, 0.1), heaviside=linear_heaviside)
    else:
        raise RuntimeError("Unknown Loss {}".format(loss))


def l_tp(gt, pt, thresh, heaviside=linear_heaviside):
    # assert gt.shape == pt.shape
    # output closer to 1 if a true positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 1 -> (inverter = false)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = tf.where(
        thresh == 0.0, 0.01, tf.where(thresh == 1.0, 0.99, thresh)
    )
    gt_t = tf.reshape(tf.repeat(gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = tf.reshape(tf.repeat(pt, thresh.shape[0]), (-1, thresh.shape[0]))
    condition = (gt_t == 0) & (pt_t >= thresh)
    xs = tf.where(condition, 1-pt_t, pt_t)
    thresholds = tf.where(condition, 1-thresh, thresh)
    return tf.reduce_sum(tf.vectorized_map(heaviside, (xs, thresholds)), axis=0)


def l_fn(gt, pt, thresh, heaviside=linear_heaviside):
    #assert gt.shape == pt.shape
    # output closer to 1 if a false negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 1 -> (inverter = true)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (inverter = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = tf.where(thresh == 0.0, 0.01, tf.where(
        thresh == 1.0, 0.99, thresh))
    gt_t = tf.reshape(tf.repeat(gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = tf.reshape(tf.repeat(pt, thresh.shape[0]), (-1, thresh.shape[0]))
    condition = (gt_t == 0) & (pt_t < thresh)
    xs = tf.where(condition, pt_t, 1-pt_t)
    thresholds = tf.where(condition, thresh, 1-thresh)
    return tf.reduce_sum(tf.vectorized_map(heaviside, (xs, thresholds)), axis=0)


def l_fp(gt, pt, thresh, heaviside=linear_heaviside):
    #assert gt.shape == pt.shape
    # output closer to 1 if a false positive, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (inverter = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (inverter = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 1 -> (inverter = false)
    #  tn: (gt == 0 and pt == 0) -> closer to 0 -> (inverter = false)
    thresh = tf.where(thresh == 0.0, 0.01, tf.where(
        thresh == 1.0, 0.99, thresh))
    gt_t = tf.reshape(tf.repeat(gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = tf.reshape(tf.repeat(pt, thresh.shape[0]), (-1, thresh.shape[0]))
    condition = (gt_t == 1) & (pt_t >= thresh)
    xs = tf.where(condition, 1-pt_t, pt_t)
    thresholds = tf.where(condition, 1-thresh, thresh)
    return tf.reduce_sum(tf.vectorized_map(heaviside, (xs, thresholds)), axis=0)


def l_tn(gt, pt, thresh, heaviside=linear_heaviside):
    #assert gt.shape == pt.shape
    # output closer to 1 if a true negative, else closer to 0
    #  tp: (gt == 1 and pt == 1) -> closer to 0 -> (invert = true)
    #  fn: (gt == 1 and pt == 0) -> closer to 0 -> (invert = false)
    #  fp: (gt == 0 and pt == 1) -> closer to 0 -> (invert = true)
    #  tn: (gt == 0 and pt == 0) -> closer to 1 -> (invert = true)
    thresh = tf.where(thresh == 0.0, 0.01, tf.where(
        thresh == 1.0, 0.99, thresh))
    gt_t = tf.reshape(tf.repeat(gt, thresh.shape[0]), (-1, thresh.shape[0]))
    pt_t = tf.reshape(tf.repeat(pt, thresh.shape[0]), (-1, thresh.shape[0]))
    condition = (gt_t == 1) & (pt_t < thresh)
    xs = tf.where(condition, pt_t, 1-pt_t)
    thresholds = tf.where(condition, thresh, 1-thresh)
    return tf.reduce_sum(tf.vectorized_map(heaviside, (xs, thresholds)), axis=0)


def tf_confusion(gt, pt, thresholds, heaviside=linear_heaviside):
    # 'tp', 'fn', 'fp', 'tn'
    tp = l_tp(gt, pt, thresholds, heaviside=heaviside)
    fn = l_fn(gt, pt, thresholds, heaviside=heaviside)
    fp = l_fp(gt, pt, thresholds, heaviside=heaviside)
    tn = l_tn(gt, pt, thresholds, heaviside=heaviside)
    return tp, fn, fp, tn


def auroc_approx(gt, pt):
    """Approximate auroc:
        - Linear interpolated Heaviside function
        - roc (11-point approximation)
        - integrate via trapezoidal rule under curve
    """
    with tf.name_scope("AurocScore"):
        thresholds = tf.range(0.1, 1, 0.1)
        tp, fn, fp, _ = tf_confusion(gt, pt, thresholds)
        precision = tp/(tp+fp+K.epsilon())
        recall = tp/(tp+fn+K.epsilon())

        # approximate area give by max precision and recall
        # this works bc/ our heaviside function approximation of
        # precision and recall approach 0, not 1 at the limit
        # maximize area
        approx_auc = tf.reduce_max(precision) * tf.reduce_max(recall)
        return 1 - approx_auc


def auprc_approx(gt, pt):
    """Approximate auprc is:
        - Linear interpolated Heaviside function
        - Single square area between max(precision) * max(recall)
    """
    with tf.name_scope("AuprcScore"):
        thresholds = tf.range(0.1, 1, 0.1)
        tp, fn, fp, _ = tf_confusion(gt, pt, thresholds)
        precision = tp/(tp+fp+K.epsilon())
        recall = tp/(tp+fn+K.epsilon())

        # approximate area give by max precision and recall
        # this works bc/ our heaviside function approximation of
        # precision and recall approach 0, not 1 at the limit
        # maximize area
        approx_auc = tf.reduce_max(precision) * tf.reduce_max(recall)
        return 1 - approx_auc


def auprc_approx_mean(gt, pt):
    """Approximate auprc is:
        - Linear interpolated Heaviside function
        - Single square area between mean(precision) * mean(recall)
    """
    with tf.name_scope("AuprcScore"):
        thresholds = tf.range(0.1, 1, 0.1)
        tp, fn, fp, _ = tf_confusion(gt, pt, thresholds)
        precision = tp/(tp+fp+K.epsilon())
        recall = tp/(tp+fn+K.epsilon())

        # approximate area give by max precision and recall
        # this works bc/ our heaviside function approximation of
        # precision and recall approach 0, not 1 at the limit
        # maximize area
        approx_auc = tf.reduce_mean(precision) * tf.reduce_mean(recall)
        return 1 - approx_auc


def mean_pr_approx(gt, pt):
    """Approximate average precision recall is:
        - Linear interpolated Heaviside function
        - average of (precision * recall) across all values
       Useful in computing AP score for object detection
    """
    with tf.name_scope("AuprcScore"):
        thresholds = tf.range(0.1, 1, 0.1)
        tp, fn, fp, _ = tf_confusion(gt, pt, thresholds)
        precision = tp/(tp+fp+K.epsilon())
        recall = tp/(tp+fn+K.epsilon())

        # approximate area give by max precision and recall
        # this works bc/ our heaviside function approximation of
        # precision and recall approach 0, not 1 at the limit
        # maximize area
        approx_apr = tf.reduce_mean(precision*recall)
        return 1 - approx_apr


def true_positive_rate(gt, pt):
    """true positive rate is precision:
        - Linear interpolated Heaviside function
        - average of (precision) across all values
    """
    with tf.name_scope("TruePositiveRateScore"):
        thresholds = tf.range(0.1, 1, 0.1)
        tp, fn, fp, _ = tf_confusion(gt, pt, thresholds)
        tpr = tp/(tp+fp+K.epsilon())
        approx_tpr = tf.reduce_mean(tpr)
        return 1 - approx_tpr


def false_positive_rate(gt, pt):
    """false positive rate is:
        - Linear interpolated Heaviside function
        - average of (fpr) across all values
    """
    with tf.name_scope("FalsePositiveRateScore"):
        thresholds = tf.range(0.1, 1, 0.1)
        tp, fn, _, tn = tf_confusion(gt, pt, thresholds)
        fpr = tp/(tp+tn+K.epsilon())
        approx_fpr = tf.reduce_mean(fpr)
        return approx_fpr


def accuracy_approx(heaviside=linear_heaviside):
    """accuracy approx is:
        - Linear interpolated Heaviside function
        - average of accuracy across all values
    """
    return accuracy_at(tf.range(0.1, 1, 0.1), heaviside=heaviside)


def accuracy_at(thresholds, heaviside):
    """Approximate F1 is:
        - Linear interpolated Heaviside function
        - Harmonic mean of precision and recall
        - Mean over a range of thresholds
    """
    def loss(gt, pt):
        with tf.name_scope("AccuracyScore"):
            tp, fn, fp, tn = tf_confusion(
                gt, pt, thresholds, heaviside=heaviside)
            accuracy = (tp+tn)/(tp+tn+fp+fn)
            approx_accuracy = tf.reduce_mean(accuracy)
            return 1 - approx_accuracy
    return loss


def mean_fb_approx_at(thresholds, b, heaviside):
    """Approximate Fb is:
        - Linear interpolated Heaviside function
        - Harmonic mean of precision and recall
        - Mean over a range of thresholds (with one at 0.5 allowed)
        - b is chosen such that recall is b times more important than precision
    """
    def loss(gt, pt):
        with tf.name_scope("F1Score"):
            tp, fn, fp, _ = tf_confusion(
                gt, pt, thresholds, heaviside=heaviside)
            precision = tp/(tp+fp+K.epsilon())
            recall = tp/(tp+fn+K.epsilon())
            fb = tf.reduce_mean((1 + tf.pow(b, 2)) * (precision * recall) /
                                (tf.pow(b, 2) * precision + recall + K.epsilon()))
            return 1 - fb
    return loss


def max_fb_approx_at(thresholds, b, heaviside):
    """Max Fb is:
        - Linear interpolated Heaviside function
        - Harmonic mean of precision and recall
        - Max over a range of thresholds
    """
    def loss(gt, pt):
        with tf.name_scope("F1Score"):
            tp, fn, fp, _ = tf_confusion(
                gt, pt, thresholds, heaviside=heaviside)
            precision = tp/(tp+fp+K.epsilon())
            recall = tp/(tp+fn+K.epsilon())
            # shape is thresholds.shape[0]
            f1s = (1 + tf.pow(b, 2)) * (precision * recall) / \
                (tf.pow(b, 2) * precision + recall + K.epsilon())
            #tf.print("f1s: ", f1s.shape)
            #idxmax = np.argmax(f1s.numpy(), axis=0)
            #max_thresh = thresholds[idxmax]
            #tf.print("max_thresh: ", max_thresh, max_thresh.shape)
            return 1 - tf.reduce_max(f1s)
    return loss


@tf.function
def roc_auc_score(y_true, y_pred):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("RocAucScore"):

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma

        masked = tf.boolean_mask(difference, difference < 0.0)

        return tf.reduce_sum(tf.pow(-masked, p))


def custom_loss_roc(layer):
    return roc_auc_score
