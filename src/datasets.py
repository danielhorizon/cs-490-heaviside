from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.datasets import make_imbalance
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix

import logging
import sklearn
import pandas as pd
import numpy as np
import scipy.io as sio


REPORTED = ['wine_quality', 'kaggle_cc_fraud']
REAL = ['kaggle_cc_fraud', 'wine_quality', ]


def wine_quality(shuffle=True):
    raw_df = pd.read_csv('./data/whitewine.csv')

    labels = raw_df['quality']
    positive_class = list(range(7, 11))
    pos, neg = 0, 0
    for label in labels:
        if label in positive_class:
            pos += 1
        else:
            neg += 1
    total = neg + pos
    logging.info('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    # Split and shuffle
    train_df, test_df = train_test_split(
        raw_df, test_size=0.2, shuffle=shuffle)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, shuffle=shuffle)

    def binarize(x): return 1 if x >= 7 else 0

    def list_binarize(array):
        for x in range(len(array)):
            array[x] = binarize(array[x])
        return array

    train_labels = list_binarize(np.array(train_df.pop('quality')))
    val_labels = list_binarize(np.array(val_df.pop('quality')))
    test_labels = list_binarize(np.array(test_df.pop('quality')))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    # Scale data
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    logging.info("whitewine mean: {}".format(train_features.mean()))
    logging.info("whitewine variance: {}".format(train_features.var()))
    logging.info("whitewine min: {}, max: {}".format(
        train_features.min(), train_features.max()))
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    logging.info('whitewine Training labels shape: {}'.format(
        train_labels.shape))
    logging.info(
        'whitewine Validation labels shape: {}'.format(val_labels.shape))
    logging.info('whitewine Test labels shape: {}'.format(test_labels.shape))
    logging.info('whitewine Training features shape: {}'.format(
        train_features.shape))
    logging.info('whitewine Validation features shape: {}'.format(
        val_features.shape))
    logging.info('whitewine Test features shape: {}'.format(
        test_features.shape))

    return {
        'train': {
            'X': train_features,
            'y': train_labels
        },
        'val': {
            'X': val_features,
            'y': val_labels
        },
        'test': {
            'X': test_features,
            'y': test_labels
        },
    }


def kaggle_cc_fraud_inverted():
    return kaggle_cc_fraud(invert=True)


def kaggle_cc_fraud(invert=False):
    '''
        https://www.kaggle.com/mlg-ulb/creditcardfraud

        Examples:
            Total: 284807
            Positive: 492 (0.17% of total)

        -- Sampled! --
        Training Examples:
            Total: 182276
            Positive: 334 (0.18% of total)

        Validation Examples:
            Total: 45569
            Positive: 78 (0.17% of total)

        Test Examples:
            Total: 56962
            Positive: 80 (0.14% of total)
    '''
    raw_df = pd.read_csv('./data/kaggle/creditcard.csv')
    if invert:
        raw_df['Class'] = raw_df['Class'].replace({0: 1, 1: 0})
    neg, pos = np.bincount(raw_df['Class'])
    total = neg + pos

    # Data balance
    logging.info('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    # Clean per the tf. example: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    cleaned_df = raw_df.copy()
    cleaned_df.pop('Time')
    # The `Amount` column covers a huge range. Convert to log-space.
    eps = 0.001  # 0 => 0.1¢
    cleaned_df['Log Amount'] = np.log(cleaned_df.pop('Amount')+eps)

    # Split and shuffle
    train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    # Form np arrays of labels and features.
    train_labels = np.array(train_df.pop('Class'))
    val_labels = np.array(val_df.pop('Class'))
    test_labels = np.array(test_df.pop('Class'))
    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    # scale
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    logging.info("kaggle mean: {}".format(train_features.mean()))
    logging.info("kaggle variance: {}".format(train_features.var()))
    logging.info("kaggle min: {}, max: {}".format(
        train_features.min(), train_features.max()))
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    logging.info('kaggle Training labels shape: {}'.format(train_labels.shape))
    logging.info('kaggle Validation labels shape: {}'.format(val_labels.shape))
    logging.info('kaggle Test labels shape: {}'.format(test_labels.shape))
    logging.info('kaggle Training features shape: {}'.format(
        train_features.shape))
    logging.info('kaggle Validation features shape: {}'.format(
        val_features.shape))
    logging.info('kaggle Test features shape: {}'.format(test_features.shape))

    return {
        'train': {
            'X': train_features,
            'y': train_labels
        },
        'val': {
            'X': val_features,
            'y': val_labels
        },
        'test': {
            'X': test_features,
            'y': test_labels
        },
    }


def mammography_inverted():
    return mammography(invert=True)


def mammography(invert=False):
    '''
        http://odds.cs.stonybrook.edu/mammography-dataset/
        Examples:
            Total: 11183
            Positive: 260 (2.32% of total)
        -- Sampled! --
        Training Examples:
            Total: 7156
            Positive: 151 (2.11% of total)
        Validation Examples:
            Total: 1790
            Positive: 48 (2.68% of total)
        Test Examples:
            Total: 2237
            Positive: 61 (2.73% of total)
    '''
    raw_df = pd.read_csv('../data/odds/mammography.csv')
    raw_df.columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'Class']

    def replace_val(x): return 0 if x == "'-1'" else 1
    _list = [replace_val(x) for x in raw_df['Class']]
    del raw_df['Class']
    raw_df['Class'] = _list

    if invert:
        raw_df['Class'] = raw_df['Class'].replace({0: 1, 1: 0})

    # Data balance
    neg, pos = np.bincount(raw_df['Class'])
    total = neg + pos
    logging.info('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    # Split and shuffle
    train_df, test_df = train_test_split(raw_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    train_labels = np.array(train_df.pop('Class'))
    val_labels = np.array(val_df.pop('Class'))
    test_labels = np.array(test_df.pop('Class'))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    # Scale data
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    logging.info("mammography mean: {}".format(train_features.mean()))
    logging.info("mammography variance: {}".format(train_features.var()))
    logging.info("mammography min: {}, max: {}".format(
        train_features.min(), train_features.max()))
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    logging.info('mammography Training labels shape: {}'.format(
        train_labels.shape))
    logging.info(
        'mammography Validation labels shape: {}'.format(val_labels.shape))
    logging.info('mammography Test labels shape: {}'.format(test_labels.shape))
    logging.info('mammography Training features shape: {}'.format(
        train_features.shape))
    logging.info('mammography Validation features shape: {}'.format(
        val_features.shape))
    logging.info('mammography Test features shape: {}'.format(
        test_features.shape))

    return {
        'train': {
            'X': train_features,
            'y': train_labels
        },
        'val': {
            'X': val_features,
            'y': val_labels
        },
        'test': {
            'X': test_features,
            'y': test_labels
        },
    }
