import tensorflow as tf
from tensorflow.python.keras.metrics import SensitivitySpecificityBase
from tensorflow.python.ops import math_ops


def get():
    return [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='roc_auc', curve='ROC'),
        tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
        F1Score(name="f1"),
        AvgF1Score(name="f1_mean"),
        MaxF1Score(name="f1_max"),
    ]


class F1Score(SensitivitySpecificityBase):
    # threshold is set to 0.5 when num_thresholds is 1
    #  see: https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/metrics.py#L1446
    def __init__(self, num_thresholds=1, name=None, dtype=None):
        self.num_thresholds = num_thresholds
        super(F1Score, self).__init__(
            value=0,
            num_thresholds=num_thresholds,
            name=name,
            dtype=dtype)

    def update_state(self, *args, **kwargs):
        # see: https://github.com/tensorflow/tensorflow/issues/30711#issuecomment-512921409
        # change is to not return a value
        super(F1Score, self).update_state(*args, **kwargs)

    def result(self):
        # Calculate precision at all the thresholds.
        precisions = math_ops.div_no_nan(
            self.true_positives, self.true_positives + self.false_positives)
        # Calculate recall at all the thresholds.
        recalls = math_ops.div_no_nan(
            self.true_positives, self.true_positives + self.false_negatives)
        # mean of f1 scores over all thresholds
        return tf.reduce_mean(2 * precisions * recalls/(precisions + recalls + tf.keras.backend.epsilon()))


class AvgF1Score(SensitivitySpecificityBase):
    def __init__(self, num_thresholds=200, name=None, dtype=None):
        self.num_thresholds = num_thresholds
        super(AvgF1Score, self).__init__(
            value=0,
            num_thresholds=num_thresholds,
            name=name,
            dtype=dtype)

    def update_state(self, *args, **kwargs):
        # see: https://github.com/tensorflow/tensorflow/issues/30711#issuecomment-512921409
        # change is to not return a value
        super(AvgF1Score, self).update_state(*args, **kwargs)

    def result(self):
        # Calculate precision at all the thresholds.
        precisions = math_ops.div_no_nan(
            self.true_positives, self.true_positives + self.false_positives)
        # Calculate recall at all the thresholds.
        recalls = math_ops.div_no_nan(
            self.true_positives, self.true_positives + self.false_negatives)
        # mean of f1 scores over all thresholds
        return tf.reduce_mean(2 * precisions * recalls/(precisions + recalls + tf.keras.backend.epsilon()))


class MaxF1Score(SensitivitySpecificityBase):
    def __init__(self, num_thresholds=200, name=None, dtype=None):
        self.num_thresholds = num_thresholds
        super(MaxF1Score, self).__init__(
            value=0,
            num_thresholds=num_thresholds,
            name=name,
            dtype=dtype)

    def update_state(self, *args, **kwargs):
        # see: https://github.com/tensorflow/tensorflow/issues/30711#issuecomment-512921409
        # change is to not return a value
        super(MaxF1Score, self).update_state(*args, **kwargs)

    def result(self):
        # Calculate precision at all the thresholds.
        precisions = math_ops.div_no_nan(
            self.true_positives, self.true_positives + self.false_positives)
        # Calculate recall at all the thresholds.
        recalls = math_ops.div_no_nan(
            self.true_positives, self.true_positives + self.false_negatives)
        # max f1 score over all thresholds
        return tf.reduce_max(2 * precisions * recalls/(precisions + recalls + tf.keras.backend.epsilon()))
