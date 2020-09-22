from setup_paths import *

import importlib
import time
import pandas as pd
import csv
import numpy as np

import tensorflow as tf
from tensorflow import keras
import sklearn

from experiment import Experiment
import datasets
import losses
import metrics

def millis():
    return int(round(time.time() * 1000))

def dataset_from_name(dataset):
    return getattr(importlib.import_module('datasets'), dataset)

class Model():

    def __init__(self, experiment):
        self.experiment = experiment
        self.args = experiment.args

    def make(self, input_shape=None, output_bias=None, name=None, loss='bce'):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        output_layer = keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)

        model = keras.Sequential([
          keras.layers.Dense(
              32, activation='relu',
              input_shape=input_shape),
          keras.layers.Dropout(0.5),
          keras.layers.Dense(
              16, activation='relu'),
          keras.layers.Dropout(0.5),
          output_layer
        ], name=name)

        optimizer = keras.optimizers.Adam(lr=1e-3)
        loss = losses.get(loss)
        model.compile(
          optimizer=optimizer,
          loss=loss,
          metrics=metrics.get())

        return model, optimizer, loss

    def callbacks(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            patience=self.args.early_stopping_patience,
            mode='min',
            restore_best_weights=True)
        return [early_stopping]

    def train_and_evaluate(self):
        print("Training and Evaluating: {}".format(self.args))
        ds = [self.args.dataset]
        if self.args.dataset == 'all':
            ds = datasets.ALL
        elif self.args.dataset == 'synthetic':
            ds = datasets.SYNTHETIC
        elif self.args.dataset == 'real':
            ds = datasets.REAL
        elif self.args.dataset == 'reported':
            ds = datasets.REPORTED
        _losses = [self.args.loss]
        if self.args.loss == 'all':
            _losses = losses.ALL
        elif self.args.loss == 'heaviside':
            _losses = losses.HEAVISIDE
        elif self.args.loss == 'lm':
            _losses = losses.LOSSES_AND_METRICS
        elif self.args.loss == 'lmp':
            _losses = losses.LOSSES_AND_METRICS_PLUS
        elif self.args.loss == 'mvs':
            _losses = losses.MEAN_VS_SINGLE
        elif self.args.loss == 'fb':
            _losses = losses.F_B
        elif self.args.loss == 'reported':
            _losses = losses.REPORTED
        elif self.args.loss == 'hs_a_r':
            _losses = losses.HEAVISIDE_AND_REPORTED
        timestamps = {
            'key':[],
            'loss':[],
            'dataset_name':[],
            'start':[],
            'end':[]
        }
        # optionally add inverted dataset
        if self.args.invert_datasets:
            _ds = []
            for d in ds:
                _ds.append(d)
                _ds.append("{}_inverted".format(d))
            ds = _ds
        # build model per each loss and load same weights
        for dataset_name in ds:
            ds = dataset_from_name(dataset_name)()
            input_shape=(ds['train']['X'].shape[-1],)

            # TODO:
            # Make a list of x-y pairs from testing data (for batching)
            test_data_points = []
            for x, y in zip(ds['test']['X'], ds['test']['y']):
                test_data_points.append([x, y])

            # init weights and load these in all other models
            weights_model, _, _ = self.make(input_shape=input_shape)
            weights_model.save_weights(self.experiment.initial_weight_path)

            if self.args.eval_batch_f1:
                allx = np.concatenate((ds['train']['X'], ds['val']['X'], ds['test']['X']))
                ally = np.concatenate((ds['train']['y'], ds['val']['y'], ds['test']['y']))

            self.experiment.models[dataset_name] = {}
            self.experiment.histories[dataset_name] = {}
            self.experiment.predictions[dataset_name] = {}
            self.experiment.results[dataset_name] = {}

            for loss in _losses:
                model, optimizer, training_loss = self.make(name="loss_{}".format(loss), input_shape=input_shape, loss=loss)
                model.load_weights(self.experiment.initial_weight_path)

                # init tensorboard, per
                #   https://stackoverflow.com/questions/61172053/tensorboard-graph-with-custom-training-loop-does-not-include-my-model/61173028#61173028
                model_tb_path = self.experiment.tensorboard_log_path.joinpath('{}_{}'.format(dataset_name, loss))
                tensorboard_callback = keras.callbacks.TensorBoard(log_dir=model_tb_path)
                tensorboard_callback.set_model(model)
                summary_writer = {
                    'train': tf.summary.create_file_writer(str(model_tb_path.joinpath('train'))),
                    'val': tf.summary.create_file_writer(str(model_tb_path.joinpath('val'))),
                    'test': tf.summary.create_file_writer(str(model_tb_path.joinpath('test'))),
                }

                self.experiment.log.info("Fitting: {} on {}".format(loss, dataset_name))
                self.experiment.predictions[dataset_name][loss] = {}
                self.experiment.results[dataset_name][loss] = {}
                timestamps['key'].append('-'.join([loss, dataset_name]))
                timestamps['loss'].append(loss)
                timestamps['dataset_name'].append(dataset_name)
                timestamps['start'].append(millis())


                if not self.args.eval_batch_f1:
                    # this is much faster if we don't need the custom loop
                    history = model.fit(
                        ds['train']['X'],
                        ds['train']['y'],
                        verbose=self.args.train_verbose,
                        batch_size=self.args.batch_size,
                        epochs=self.args.epochs,
                        callbacks=self.callbacks(),
                        validation_data=(ds['val']['X'], ds['val']['y']))

                else:
                    # custom loop
                    best_loss = None
                    patience = self.args.early_stopping_patience
                    early_stopping = False
                    train_labels = ds['train']['y']
                    train_dataset = tf.data.Dataset.from_tensor_slices((ds['train']['X'], train_labels))
                    train_set = train_dataset.batch(self.args.batch_size)

                    df_dict = {
                        'loss': [],
                        'epoch': [],
                        'batch': [],
                        'train_f1': [],
                        'all_f1': [],
                        'delta_f1': [],
                    }

                    print(f"Going through the dataset {self.args.epochs} times.\n"+("="*30))
                    for epoch in range(self.args.epochs):
                        if early_stopping:
                            print("Early Stopping at Epoch {}/{}".format(epoch, self.args.epochs))
                            break
                        # Looping through each bath
                        batch = 0
                        epoch_best_loss = None
                        for x, y in train_set:
                            with tf.GradientTape() as tape:
                                # we must specify training=True to ensure dropout is properly applied
                                predictions = model(x, training=True)
                                loss_val = training_loss(y, predictions)
                            gradients = tape.gradient(loss_val, model.trainable_variables)
                            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                            if epoch_best_loss is None or epoch_best_loss < loss_val:
                                epoch_best_loss = loss_val

                            # early stopping logic
                            if self.args.eval_batch_f1:
                                # TODO: test a range of thresholds
                                threshold = 0.5
                                # train set
                                train_predictions = tf.where(model.predict(x) > threshold, 1, 0)
                                train_f1 = sklearn.metrics.f1_score(y, train_predictions)

                                # all data
                                all_predictions = tf.where(model.predict(allx) > threshold, 1, 0)
                                all_f1 = sklearn.metrics.f1_score(ally, all_predictions)

                                # should be negative when all f1 is greater than train f1
                                delta_f1 = all_f1 - train_f1

                                df_dict['loss'].append(loss)
                                df_dict['epoch'].append(epoch)
                                df_dict['batch'].append(batch)
                                df_dict['train_f1'].append(train_f1)
                                df_dict['all_f1'].append(all_f1)
                                df_dict['delta_f1'].append(delta_f1)

                            batch += 1

                        # check early stopping per epoch
                        patience -= 1
                        if best_loss is None or best_loss > epoch_best_loss:
                            best_loss = epoch_best_loss
                            patience = self.args.early_stopping_patience
                        if patience <= 0:
                            early_stopping = True

                        # write to tensorboard
                        for split in ['train', 'val', 'test']:
                            evaluation = model.evaluate(ds[split]['X'], ds[split]['y'], batch_size=self.args.batch_size, verbose=0)
                            with summary_writer[split].as_default():
                                tf.summary.scalar('loss', epoch_best_loss, step=epoch)
                                tf.summary.scalar('patience', epoch_best_loss, step=epoch)
                                for name, value in zip(model.metrics_names, evaluation):
                                    tf.summary.scalar(name, value, step=epoch)

                        print("(Dataset: {}, Loss: {}) Epoch [{}]: val_loss: {}".format(dataset_name, loss, epoch, epoch_best_loss))

                    if self.args.eval_batch_f1:
                        # Write the data in a csv:
                        f1_storage_path = str(self.experiment.path.joinpath("f1_per_batch_loss-{}_dataset-{}_batch_size-{}.csv".format(loss, dataset_name, self.args.batch_size)))
                        df = pd.DataFrame.from_dict(df_dict)
                        df.to_csv(f1_storage_path)

                timestamps['end'].append(millis())

                for split in ['train', 'test']:
                    self.experiment.log.info("{} on {}:{} evaluation".format(loss, dataset_name, split))
                    self.experiment.predictions[dataset_name][loss][split] = model.predict(ds[split]['X'], batch_size=self.args.batch_size)
                    evaluated = model.evaluate(ds[split]['X'], ds[split]['y'], batch_size=self.args.batch_size, verbose=0)
                    self.experiment.results[dataset_name][loss][split] = {}

                    # write to tensorboard and save to pkl in slef.experiment.results
                    for name, value in zip(model.metrics_names, evaluated):
                        self.experiment.log.info("{}: {}".format(name, value))
                        self.experiment.results[dataset_name][loss][split][name] = value

                model_path = str(self.experiment.path.joinpath("model_{}_{}.h5".format(dataset_name, loss)))
                timing_path = str(self.experiment.path.joinpath("model_{}_{}_timing.pkl".format(dataset_name, loss)))
                pd.DataFrame.from_dict(timestamps).to_pickle(timing_path)
                tf.keras.models.save_model(model, model_path,
                    overwrite=True, include_optimizer=True, save_format=None,
                    signatures=None, options=None
                )
                self.experiment.models[dataset_name][loss] = model_path

        self.experiment.save()
