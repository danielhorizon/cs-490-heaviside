#!/usr/bin/env python3
#
# Nathan Tsoi © 2020
#
# Installation:
#   pip3 install numpy pandas scipy sklearn tensorflow-gpu
#

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.metrics import SensitivitySpecificityBase
from tensorflow.python.ops import math_ops

# Set to wherever you put `mammography.mat`
DATA_PATH = 'data/odds'

def mammography():
    # http://odds.cs.stonybrook.edu/mammography-dataset/
    D = sio.loadmat('{}/mammography.mat'.format(DATA_PATH))
    X = D['X'].astype(np.float32)
    y = D['y']
    raw_df = pd.DataFrame(X)
    raw_df['y'] = D['y']
    raw_df.columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'Class']
    raw_df

    # Data balance
    neg, pos = np.bincount(raw_df['Class'])
    total = neg + pos

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
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

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

class F1Score(SensitivitySpecificityBase):
    # threshold=0.5 if num_thresholds is 1
    def __init__(self, num_thresholds=1, name=None, dtype=None):
        self.num_thresholds = num_thresholds
        super(F1Score, self).__init__(
            value=0,
            num_thresholds=num_thresholds,
            name=name,
            dtype=dtype)

    def update_state(self, *args, **kwargs):
        super(F1Score, self).update_state(*args, **kwargs)

    def result(self):
        precisions = math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_positives)
        recalls = math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_negatives)
        return tf.reduce_mean(2 * precisions * recalls/(precisions + recalls + tf.keras.backend.epsilon()))

def main():
    batch_size=2048
    epochs=5000
    patience=100
    learning_rate=1e-3

    ds = mammography()

    model = keras.Sequential([
      keras.layers.Dense(
          32, activation='relu',
          input_shape=ds['train']['X'].shape),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(
          16, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
      optimizer=keras.optimizers.Adam(lr=learning_rate),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=[F1Score(name="f1")])

    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        verbose=1,
        patience=patience,
        mode='min',
        restore_best_weights=True))

    history = model.fit(
        ds['train']['X'],
        ds['train']['y'],
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(ds['val']['X'], ds['val']['y']))

    split = 'test'
    model.evaluate(ds[split]['X'], ds[split]['y'], batch_size=batch_size, verbose=1)

if __name__ == '__main__':
    main()
