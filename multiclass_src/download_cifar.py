import os 
import random
import numpy as np

from random import sample 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# custom imports 
from cifar_helper import * 

# portions of this are cited from: 
# https://raw.githubusercontent.com/Hvass-Labs/TensorFlow-Tutorials/master/download.py

def generate_random(n):
    random_idx = []
    for i in range(n):
        random_idx.append(random.randint(0, 999))
    print("{}".format(len(random_idx)))
    return random_idx


def get_idx_list(arr): 
    class_nine = [] 
    for i in range(len(arr)):
        if arr[i] == 9:
            # append index where this is 
            class_nine.append(i)
    print("Class 9: {}".format(len(class_nine)))
    return class_nine


def convert_shape(x): 
    return x.reshape([1,3,32,32])


def create_imbalance_train_v2(images, labels, seed):
    class_count = np.unique(labels, return_counts=True)[1]
    # gets us indices of all class 9's - 5000 indices
    class_nine_idx = get_idx_list(labels)
    print("LENGTH OF THIS CLASS:{}".format(len(class_nine_idx)))
    subset_amt = int(len(class_nine_idx) * 0.8)

    remove_indices = sample(class_nine_idx, subset_amt)
    imb_images = [i for j, i in enumerate(images) if j not in remove_indices]
    imb_labels = [i for j, i in enumerate(labels) if j not in remove_indices]

    # getting the 1000 indices that are left that are class 9
    class_count = np.unique(imb_labels, return_counts=True)[1]
    print("Class Count: {}".format(class_count))
    class_nine_leftover = get_idx_list(imb_labels)
    print("Size of leftover: {}".format(len(class_nine_leftover)))

    # sample from the 1000 indices 4000 times
    random.seed(seed)
    list_4800_indices = random.choices(class_nine_leftover, k=subset_amt)

    for idx in list_4800_indices:
        imb_images.append(imb_images[idx])
        imb_labels.append(imb_labels[idx])

    class_count = np.unique(imb_labels, return_counts=True)[1]
    # need to split into validation as well
    print("Class Count: {}".format(class_count))
    return imb_images, imb_labels



def load_imb_data_v2(seed):
    """ Loads imbalanced data (80-20 split on class 9, and sampling again from 20%). 
    """
    maybe_download_and_extract()
    train_imgs, train_cls, _ = load_training_data()
    test_imgs, test_cls, _ = load_test_data()

    imgs = np.vstack((train_imgs, test_imgs))
    print("dataset shape: {}".format(imgs.shape))
    cls = np.hstack((train_cls, test_cls))
    print("labels shape: {}".format(cls.shape))

    # reshaping images.
    images = []
    for i in range(len(imgs)):
        images.append(imgs[i].reshape([3, 32, 32]))
    images = np.array(images)
    

    # creating imbalance, and oversampling.
    X, y = create_imbalance_train_v2(images=images, labels=cls, seed=seed)

    # https://stackoverflow.com/questions/58955816/difference-between-shuffle-and-random-state-in-train-test-split
    # if you run the code again with the same random_state, the output will always remain the same.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed)  # 0.25 x 0.8 = 0.2


    print("Shape of train: {}".format(X_train[0].shape))
    print("Shape of test: {}".format(X_test[0].shape))
    print("Shape of val: {}".format(X_valid[0].shape))
    print("Size of train labels: {}".format(len(y_train)))
    print("Size of valid labels: {}".format(len(y_valid)))
    print("Size of test labels: {}".format(len(y_test)))

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_valid = np.array(X_valid)

    # Adding in scaling
    # https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_valid = scaler.transform(
        X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    print("Shape of train: {}".format(X_train[0].shape))
    print("Shape of test: {}".format(X_test[0].shape))
    print("Shape of val: {}".format(X_valid[0].shape))

    return {
        'train': {
            'X': X_train,
            'y': y_train
        },
        'val': {
            'X': X_valid,
            'y': y_valid
        },
        'test': {
            'X': X_test,
            'y': y_test
        },
    }
