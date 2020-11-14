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
    print("Size of leftover: {}".format(len(class_nine_leftover)))
    # sample from the 1000 indices 4000 times
    random.seed(1000)
    list_4000_indices = random.choices(class_nine_leftover, k=4000)

    for idx in list_4000_indices:
        imb_images.append(imb_images[idx])
        imb_labels.append(imb_labels[idx])
    
    class_count = np.unique(imb_labels, return_counts=True)[1]
    # need to split into validation as well
    print("Class Count: {}".format(class_count))
    return imb_images, imb_labels


def convert_shape(x): 
    return x.reshape([1,3,32,32])


def load_imb_data(seed):
    """ Loads imbalanced data (80-20 split on class 9, and sampling again from 20%). 
    """
    maybe_download_and_extract()
    imgs, cls, _ = load_training_data()

    # reshaping images. 
    images = [] 
    for i in range(len(imgs)): 
        images.append(imgs[i].reshape([3,32,32]))
    images = np.array(images)

    # creating imbalance, and oversampling. 
    X, y = create_imbalance_train(images, labels=cls)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, shuffle=True, random_state=seed)
    X_test, y_test, _ = load_test_data()

    X_test_reshaped = [] 
    for i in range(len(X_test)):
        X_test_reshaped.append(X_test[i].reshape([3, 32, 32]))
    X_test_reshaped = np.array(X_test_reshaped)

    print("Shape of train: {}".format(X_train[0].shape))
    print("Shape of test: {}".format(X_test[0].shape))
    print("Shape of val: {}".format(X_valid[0].shape))
    print("Size of train labels: {}".format(len(y_train)))
    print("Size of valid labels: {}".format(len(y_valid)))
    print("Size of test labels: {}".format(len(y_test)))

    X_train = np.array(X_train)
    X_test = np.array(X_test_reshaped)
    X_valid = np.array(X_valid)

    # Adding in scaling 
    # https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_valid = scaler.transform(
        X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
    X_test = scaler.transform(
        X_test_reshaped.reshape(-1, X_test_reshaped.shape[-1])).reshape(X_test_reshaped.shape)
    
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
    

if __name__=="__main__":
    load_imb_data()   
