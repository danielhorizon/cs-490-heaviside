from setup_paths import *

REPORTED = ['cocktailparty', 'mammography', 'kaggle_cc_fraud', 'uci_adult']
REAL = ['cocktailparty', 'mammography', 'kaggle_cc_fraud', 'uci_adult', 'titanic', 'winequality', 'robotabuse']
SYNTHETIC = ['synthetic', 'synthetic_33', 'synthetic_05']
ALL = REAL + SYNTHETIC

import scipy.io as sio

import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_blobs
from imblearn.datasets import make_imbalance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from collections import Counter

import logging

def robotabuse_inverted():
    return robotabuse(invert=True)

def robotabuse(shuffle=True, invert=False):
    raw_df = pd.read_csv(DATA_PATH.joinpath('robotabuse/agg_df.csv'))
    drop_cols = ["rel_timestamp", "screen", "participant"]
    raw_df = raw_df.drop(columns=drop_cols)

    # Make labels 0 or 1
    numerize = lambda x: 1 if x == "sad" else 0
    raw_df['condition'] = raw_df['condition'].apply(numerize)
    if invert:
        raw_df['condition'] = raw_df['condition'].replace({0:1, 1:0})

    neg, pos = np.bincount(raw_df['condition'])
    total = neg + pos
    logging.info('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    # Split and shuffle
    train_df, test_df = train_test_split(raw_df, test_size=0.2, shuffle=shuffle)
    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=shuffle)

    train_labels = np.array(train_df.pop('condition'))
    val_labels = np.array(val_df.pop('condition'))
    test_labels = np.array(test_df.pop('condition'))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    # Scale data
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    logging.info("robotabuse mean: {}".format(train_features.mean()))
    logging.info("robotabuse variance: {}".format(train_features.var()))
    logging.info("robotabuse min: {}, max: {}".format(train_features.min(), train_features.max()))
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    logging.info('robotabuse Training labels shape: {}'.format(train_labels.shape))
    logging.info('robotabuse Validation labels shape: {}'.format(val_labels.shape))
    logging.info('robotabuse Test labels shape: {}'.format(test_labels.shape))
    logging.info('robotabuse Training features shape: {}'.format(train_features.shape))
    logging.info('robotabuse Validation features shape: {}'.format(val_features.shape))
    logging.info('robotabuse Test features shape: {}'.format(test_features.shape))

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

def cocktailparty_inverted():
    return cocktailparty(invert=True)

def cocktailparty(shuffle=True, invert=False):
    '''
        timestamp:
        ij (id_1, id_2): ids of diad member
        4 features of diad 1:
        4 features of diad 2:
        4 x 4 features of other people
        [last value]: label

        (already done) preprocessing:
            center x,y,theta relative to the diad
            theta is cos() or sin()
            randomly swapped order of the diads
            [one more augmentation step] could flip vertically after centering, if results are bad, could add this

        try: random split
            - if too high, split by time: (train, val, test)
            - if too high, split by time: (test, val, train) [harder]

        Examples:
            Total: 4800
            Positive: 1454 (30.29% of total)
    '''
    raw_df = pd.read_csv(DATA_PATH.joinpath('cocktailparty/cp_binary.csv'))
    drop_cols = ["stamp","sample_no","swapped","person1_index","person2_index","person1_id","person2_id","dyad1_y","dyad2_y"]
    raw_df = raw_df.drop(columns=drop_cols)

    # note, this dataset is inverted by default
    # invert pos/neg. otherwise accuracy score is better b/c f1-score doesn't take true negatives into account
    if not invert:
        raw_df['target'] = raw_df['target'].replace({0:1, 1:0})

    neg, pos = np.bincount(raw_df['target'])
    total = neg + pos
    logging.info('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    # Split and shuffle
    train_df, test_df = train_test_split(raw_df, test_size=0.2, shuffle=shuffle)
    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=shuffle)

    train_labels = np.array(train_df.pop('target'))
    val_labels = np.array(val_df.pop('target'))
    test_labels = np.array(test_df.pop('target'))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    # Scale data
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    logging.info("cocktailparty mean: {}".format(train_features.mean()))
    logging.info("cocktailparty variance: {}".format(train_features.var()))
    logging.info("cocktailparty min: {}, max: {}".format(train_features.min(), train_features.max()))
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    logging.info('cocktailparty Training labels shape: {}'.format(train_labels.shape))
    logging.info('cocktailparty Validation labels shape: {}'.format(val_labels.shape))
    logging.info('cocktailparty Test labels shape: {}'.format(test_labels.shape))
    logging.info('cocktailparty Training features shape: {}'.format(train_features.shape))
    logging.info('cocktailparty Validation features shape: {}'.format(val_features.shape))
    logging.info('cocktailparty Test features shape: {}'.format(test_features.shape))

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

def winequality(shuffle=True):
    raw_df = pd.read_csv(DATA_PATH.joinpath('winequality/whitewine.csv'), sep=";")

    # # print(f"\n\n{raw_df[0:4]}\n\n")
    # print("\n\n"+raw_df['quality'][0:4]+"\n\n")
    # quit()

    labels = raw_df['quality']
    positive_class = list(range(7,11))
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
    train_df, test_df = train_test_split(raw_df, test_size=0.2, shuffle=shuffle)
    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=shuffle)

    binarize = lambda x: 1 if x >= 7 else 0
    def list_binarize(array):
        for x in range(len(array)):
            array[x] = binarize(array[x])
        return array
    # list_binarize(array) = lambda array: np.array([binarize(x) for x in array])

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
    logging.info("whitewine min: {}, max: {}".format(train_features.min(), train_features.max()))
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    logging.info('whitewine Training labels shape: {}'.format(train_labels.shape))
    logging.info('whitewine Validation labels shape: {}'.format(val_labels.shape))
    logging.info('whitewine Test labels shape: {}'.format(test_labels.shape))
    logging.info('whitewine Training features shape: {}'.format(train_features.shape))
    logging.info('whitewine Validation features shape: {}'.format(val_features.shape))
    logging.info('whitewine Test features shape: {}'.format(test_features.shape))

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
    raw_df = pd.read_csv(DATA_PATH.joinpath('odds/mammography.csv'))
    raw_df.columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'Class']

    replace_val = lambda x: 0 if x == "'-1'" else 1
    _list = [replace_val(x) for x in raw_df['Class']]
    del raw_df['Class']
    raw_df['Class'] = _list

    if invert:
        raw_df['Class'] = raw_df['Class'].replace({0:1, 1:0})

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
    logging.info("mammography min: {}, max: {}".format(train_features.min(), train_features.max()))
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    logging.info('mammography Training labels shape: {}'.format(train_labels.shape))
    logging.info('mammography Validation labels shape: {}'.format(val_labels.shape))
    logging.info('mammography Test labels shape: {}'.format(test_labels.shape))
    logging.info('mammography Training features shape: {}'.format(train_features.shape))
    logging.info('mammography Validation features shape: {}'.format(val_features.shape))
    logging.info('mammography Test features shape: {}'.format(test_features.shape))

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
    raw_df = pd.read_csv(DATA_PATH.joinpath('kaggle/creditcard.csv'))
    if invert:
        raw_df['Class'] = raw_df['Class'].replace({0:1, 1:0})
    neg, pos = np.bincount(raw_df['Class'])
    total = neg + pos

    # Data balance
    logging.info('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    # Clean per the tf. example: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    cleaned_df = raw_df.copy()
    cleaned_df.pop('Time')
    # The `Amount` column covers a huge range. Convert to log-space.
    eps=0.001 # 0 => 0.1¢
    cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)

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
    logging.info("kaggle min: {}, max: {}".format(train_features.min(), train_features.max()))
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    logging.info('kaggle Training labels shape: {}'.format(train_labels.shape))
    logging.info('kaggle Validation labels shape: {}'.format(val_labels.shape))
    logging.info('kaggle Test labels shape: {}'.format(test_labels.shape))
    logging.info('kaggle Training features shape: {}'.format(train_features.shape))
    logging.info('kaggle Validation features shape: {}'.format(val_features.shape))
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

def uci_adult_inverted():
    return uci_adult(invert=True)

def uci_adult(invert=False):
    '''
        https://archive.ics.uci.edu/ml/datasets/Adult

        Examples:
            Total: 48842
            Positive: 11687 (23.93% of total)

        -- Sampled! --
        Training Examples:
            Total: 26048
            Positive: 6245 (23.97% of total)

        Validation Examples:
            Total: 6513
            Positive: 1596 (24.50% of total)

        Test Examples:
            Total: 16281
            Positive: 3846 (23.62% of total)
    '''
    adult_columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','label']
    # drop fnlwgt column
    drop_cols = ['fnlwgt']
    adult_train_df = pd.read_csv(DATA_PATH.joinpath('uci/adult.data'), names=adult_columns)
    adult_train_df = adult_train_df.drop(columns=drop_cols)
    adult_train_df['split'] = 'train'
    adult_test_df = pd.read_csv(DATA_PATH.joinpath('uci/adult.test'), names=adult_columns)
    # drop fnlwgt column
    adult_test_df = adult_test_df.drop(columns=drop_cols)
    adult_test_df['split'] = 'test'
    # drop (empty) first row
    adult_test_df = adult_test_df.drop(adult_test_df.index[0])
    adult_df = pd.concat([adult_train_df, adult_test_df], ignore_index=True)
    # clean labels
    adult_df['label'] = adult_df['label'].str.replace('.', '', regex=False)
    adult_df['label'].unique()
    if invert:
        raw_df['label'] = raw_df['label'].replace({0:1, 1:0})
    # Convert categorical fields to ints
    for col_name in [c for c in adult_df.columns if c not in ['split']]:
        if(adult_df[col_name].dtype == 'object'):
            adult_df[col_name]= adult_df[col_name].astype('category')
            adult_df[col_name] = adult_df[col_name].cat.codes
    # split
    train_df = adult_df[adult_df['split'] == 'train']
    train_df = train_df.drop(columns=['split'])
    test_df = adult_df[adult_df['split'] == 'test']
    test_df = test_df.drop(columns=['split'])
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    # Form np arrays of labels and features.
    train_labels = np.array(train_df.pop('label'))
    val_labels = np.array(val_df.pop('label'))
    test_labels = np.array(test_df.pop('label'))
    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)
    # scale
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    logging.info("uci mean: {}".format(train_features.mean()))
    logging.info("uci variance: {}".format(train_features.var()))
    logging.info("uci min: {}, max: {}".format(train_features.min(), train_features.max()))
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    logging.info('uci Training labels shape: {}'.format(train_labels.shape))
    logging.info('uci Validation labels shape: {}'.format(val_labels.shape))
    logging.info('uci Test labels shape: {}'.format(test_labels.shape))
    logging.info('uci Training features shape: {}'.format(train_features.shape))
    logging.info('uci Validation features shape: {}'.format(val_features.shape))
    logging.info('uci Test features shape: {}'.format(test_features.shape))

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

def titanic_inverted():
    return titanic(invert=True)

def titanic(invert=False):
    '''
    https://www.kaggle.com/c/titanic/data
    only train is labeled, so using this for train/val/test

    Examples:
        Total: 891
        Positive: 342 (38.38% of total)

    -- Sampled! --
    Training Examples:
        Total: 569
        Positive: 220 (38.66% of total)

    Validation Examples:
        Total: 143
        Positive: 59 (41.26% of total)

    Test Examples:
        Total: 179
        Positive: 63 (35.20% of total)
    '''
    df = pd.read_csv(DATA_PATH.joinpath('kaggle/titanic/train.csv'))
    for col_name in [c for c in df.columns if c not in ['split']]:
        if(df[col_name].dtype == 'object'):
            df[col_name]= df[col_name].astype('category')
            df[col_name] = df[col_name].cat.codes

    df = df.fillna(value={'Age':-1})

    if invert:
        raw_df['Survived'] = raw_df['Survived'].replace({0:1, 1:0})

    neg, pos = np.bincount(df['Survived'])
    total = neg + pos
    logging.info('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    # Form np arrays of labels and features.
    train_labels = np.array(train_df.pop('Survived'))
    val_labels = np.array(val_df.pop('Survived'))
    test_labels = np.array(test_df.pop('Survived'))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    neg, pos = np.bincount(train_labels)
    total = neg + pos
    logging.info('Training Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    neg, pos = np.bincount(val_labels)
    total = neg + pos
    logging.info('Validation Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    neg, pos = np.bincount(test_labels)
    total = neg + pos
    logging.info('Testing Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    logging.info("titanic mean: {}".format(train_features.mean()))
    logging.info("titanic variance: {}".format(train_features.var()))
    logging.info("titanic min: {}, max: {}".format(train_features.min(), train_features.max()))
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    logging.info('titanic Training labels shape: {}'.format(train_labels.shape))
    logging.info('titanic Validation labels shape: {}'.format(val_labels.shape))
    logging.info('titanic Test labels shape: {}'.format(test_labels.shape))
    logging.info('titanic Training features shape: {}'.format(train_features.shape))
    logging.info('titanic Validation features shape: {}'.format(val_features.shape))
    logging.info('titanic Test features shape: {}'.format(test_features.shape))

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

def _synthetic(n_samples=10000, balance=None, random_state=42, invert=False):
    X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=2, n_features=3, cluster_std=10)

    def ratio_func(y, multiplier, minority_class):
        target_stats = Counter(y)
        return {minority_class: int(multiplier * target_stats[minority_class])}

    if balance is not None:
        X, y = make_imbalance(X, y,
                              sampling_strategy=ratio_func,
                              **{"multiplier": balance,
                                 "minority_class": 1},
                              random_state=random_state)
    if invert:
        y = 1 - y

    neg, pos = np.bincount(y)
    total = neg + pos
    logging.info('Synthetic {} Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        balance, total, pos, 100 * pos / total))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    neg, pos = np.bincount(y_train)
    total = neg + pos
    logging.info('Synthetic {} Training Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        balance, total, pos, 100 * pos / total))

    neg, pos = np.bincount(y_val)
    total = neg + pos
    logging.info('Synthetic {} Validation Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        balance, total, pos, 100 * pos / total))

    total = neg + pos
    logging.info('Synthetic {} Testing Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        balance, total, pos, 100 * pos / total))


    scaler = StandardScaler()
    train_features = scaler.fit_transform(X_train)
    logging.info("synthetic {} mean: {}".format(balance, X_train.mean()))
    logging.info("synthetic {} variance: {}".format(balance, X_train.var()))
    logging.info("synthetic {} min: {}, max: {}".format(balance, X_train.min(), X_train.max()))
    val_features = scaler.transform(X_val)
    test_features = scaler.transform(X_test)

    logging.info('synthetic {} Training labels shape: {}'.format(balance, y_train.shape))
    logging.info('synthetic {} Validation labels shape: {}'.format(balance, y_val.shape))
    logging.info('synthetic {} Test labels shape: {}'.format(balance, y_test.shape))
    logging.info('synthetic {} Training features shape: {}'.format(balance, train_features.shape))
    logging.info('synthetic {} Validation features shape: {}'.format(balance, val_features.shape))
    logging.info('synthetic {} Test features shape: {}'.format(balance, test_features.shape))

    return {
        'train': {
            'X': train_features,
            'y': y_train
        },
        'val': {
            'X': val_features,
            'y': y_val
        },
        'test': {
            'X': test_features,
            'y': y_test
        },
    }

def synthetic():
    return _synthetic(n_samples=10000)

def synthetic_33():
    '''
    Synthetic 0.5 Examples:
        Total: 7500
        Positive: 2500 (33.33% of total)
    '''
    return _synthetic(n_samples=10000, balance=0.5)

def synthetic_20():
    '''
    Synthetic 0.25 Examples:
        Total: 6250
        Positive: 1250 (20.00% of total)
    '''
    return _synthetic(n_samples=10000, balance=0.25)

def synthetic_05():
    '''
    Synthetic 0.05 Examples:
        Total: 5250
        Positive: 250 (4.76% of total)
    '''
    return _synthetic(n_samples=10000, balance=0.05)

# aliases for t16
def synthetic_50():
    return synthetic_33()

def synthetic_25():
    return synthetic_20()
