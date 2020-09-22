from metrics import AvgF1Score, MaxF1Score

import numpy as np
import pandas as pd

import importlib
import sklearn

def dataset_from_name(dataset):
    return getattr(importlib.import_module('datasets'), dataset)

def test_eval(dataset_name, model_path, loss, pickle_path, args):
    # don't load tf until we're in a forked thread
    import tensorflow as tf
    gpu_devices = tf.config.list_physical_devices('GPU')
    if not len(gpu_devices):
        raise RuntimeError("no GPU in: {}".format(gpu_devices))
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    import losses
    from tensorflow.keras import backend as K

    results_df_dict = {'dataset':[], 'loss': [], 'threshold': [], 'f1':[], 'accuracy': [], 'precision': [], 'recall': [],
            'ap': [], 'pr_auc': [], 'roc_auc': [], 'tpr': [], 'fpr': [], 'tn': [], 'fp': [], 'fn': [], 'tp': []}
    # problem w/ SavedModel format
    ## https://github.com/tensorflow/tensorflow/issues/33646#issuecomment-566433261
    print("LOADING: {}".format(model_path))
    model = tf.keras.models.load_model(model_path, custom_objects={"avg_f1_score": AvgF1Score, "max_f1_score": MaxF1Score}, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=losses.get(loss))
    ds = dataset_from_name(dataset_name)()
    y = model.predict(ds['test']['X'], batch_size=args.batch_size)
    thresholds = np.arange(0, 1, 0.001)
    for t in thresholds:
        pt = (y >= t).reshape(-1)
        #print('GT SHAPE', ds['test']['y'].shape)
        #print('PT SHAPE', pt.shape)
        f1 = sklearn.metrics.f1_score(ds['test']['y'], pt)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(ds['test']['y'], pt).ravel()
        precision = sklearn.metrics.precision_score(ds['test']['y'], pt)
        recall = sklearn.metrics.recall_score(ds['test']['y'], pt)
        accuracy = sklearn.metrics.accuracy_score(ds['test']['y'], pt)
        ap = sklearn.metrics.average_precision_score(ds['test']['y'], pt)
        m = tf.keras.metrics.AUC(name='pr_auc', curve='PR')
        _ = m.update_state(ds['test']['y'], pt)
        pr_auc = m.result().numpy()
        roc_auc = sklearn.metrics.roc_auc_score(ds['test']['y'], pt)
        results_df_dict['dataset'].append(dataset_name)
        results_df_dict['loss'].append(loss)
        results_df_dict['threshold'].append(t)
        results_df_dict['f1'].append(f1)
        results_df_dict['accuracy'].append(accuracy)
        results_df_dict['precision'].append(precision)
        results_df_dict['recall'].append(recall)
        results_df_dict['ap'].append(ap)
        results_df_dict['pr_auc'].append(pr_auc)
        results_df_dict['roc_auc'].append(roc_auc)
        results_df_dict['tpr'].append(tp/(tp+fp+K.epsilon()))
        results_df_dict['fpr'].append(tp/(tp+tn+K.epsilon()))
        results_df_dict['tn'].append(tn)
        results_df_dict['fp'].append(fp)
        results_df_dict['fn'].append(fn)
        results_df_dict['tp'].append(tp)
    print("SAVING {}".format(pickle_path))
    pd.DataFrame.from_dict(results_df_dict).to_pickle(pickle_path)
    return pickle_path

