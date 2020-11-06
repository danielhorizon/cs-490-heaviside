import os 
import random
import numpy as np

from random import sample 
from sklearn.model_selection import train_test_split

from cifar_helper import * 

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


# imbalance for train data 
def create_imbalance_train(images, labels): 
    print("Length of Train: {}".format(len(images)))
    print("Length of Labels: {}".format(len(labels)))

    class_count = np.unique(labels, return_counts=True)[1]
    # gets us indices of all class 9's - 5000 indices 
    class_nine_idx = get_idx_list(labels)       
    # choosing 4000 of these indices to remove from images and labels 
    remove_indices = sample(class_nine_idx, 4000)
    imb_images = [i for j, i in enumerate(images) if j not in remove_indices]
    imb_labels = [i for j, i in enumerate(labels) if j not in remove_indices]

    print("Length of IMB Train: {}".format(len(imb_images)))
    print("Length of IMB Labels: {}".format(len(imb_labels)))

    # getting the 1000 indices that are left that are class 9 
    class_count = np.unique(imb_labels, return_counts=True)[1]
    print("Class Count: {}".format(class_count))
    class_nine_leftover = get_idx_list(imb_labels)
    list_4000_indices = random.choices(class_nine_leftover, k=4000)

    for idx in list_4000_indices:
        imb_images.append(imb_images[idx])
        imb_labels.append(imb_labels[idx])
    
    class_count = np.unique(imb_labels, return_counts=True)[1]
    # need to split into validation as well
    print("Class Count: {}".format(class_count))
    return imb_images, imb_labels


def load_imb_data():
    """ Loads imbalanced data (80-20 split on class 9, and sampling again from 20%). 
    """
    maybe_download_and_extract()
    images, cls, _ = load_training_data()

    X, y = create_imbalance_train(images, labels=cls)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=1)
    X_test, y_test, _ = load_test_data()
    print("Size of train: {}".format(len(y_train)))
    print("Size of valid: {}".format(len(y_valid)))
    print("Size of test: {}".format(len(y_test)))

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
    
