import os 
import os.path
import gzip
import pickle
import numpy as np
import random 
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def loadX(fnimg):
	f = gzip.open(fnimg, 'rb')
	f.read(16)
	return np.frombuffer(f.read(), dtype=np.uint8).reshape((-1, 28*28))
    
def loadY(fnlabel):
	f = gzip.open(fnlabel, 'rb')
	f.read(8)
	return np.frombuffer(f.read(), dtype=np.uint8)
def get_idx_list(arr):
    class_nine = []
    for i in range(len(arr)):
        if arr[i] == 9:
            # append index where this is
            class_nine.append(i)
    print("Class 9: {}".format(len(class_nine)))
    return class_nine
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_imbalance_train(images, labels, seed):
    set_seed(seed)
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
    list_4800_indices = random.choices(class_nine_leftover, k=subset_amt)

    for idx in list_4800_indices:
        imb_images.append(imb_images[idx])
        imb_labels.append(imb_labels[idx])

    class_count = np.unique(imb_labels, return_counts=True)[1]
    # need to split into validation as well
    print("Class Count: {}".format(class_count))
    return imb_images, imb_labels


def load_entire_dataset(seed):
    train_X = loadX("/app/timeseries/multiclass_src/data/MNIST/train-images-idx3-ubyte.gz")
    train_y = loadY("/app/timeseries/multiclass_src/data/MNIST/train-labels-idx1-ubyte.gz")

    test_X = loadX("/app/timeseries/multiclass_src/data/MNIST/t10k-images-idx3-ubyte.gz")
    test_y = loadY("/app/timeseries/multiclass_src/data/MNIST/t10k-labels-idx1-ubyte.gz")

    print(train_X[0].shape)
    # combine it all into one big dataset 
    imgs = np.vstack((train_X, test_X))
    print("dataset shape: {}".format(imgs.shape))
    cls = np.hstack((train_y, test_y))
    print("labels shape: {}".format(cls.shape))

    images = [] 
    # convert values to being between 0 and 1
    for i in range(len(imgs)): 
        images.append(imgs[i] / 255.0)
    images = np.array(images)
    print(images[0].shape)
    
    imgs = np.array(imgs)
    cls = np.array(cls)
    
    # creating imbalance in the dataset. 
    X, y = create_imbalance_train(images=imgs, labels=cls, seed=seed)

    # do the splits, and imbalance 
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

    # then use Scaler() 
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


if __name__ == "__main__":
    test = load_entire_dataset(2) 
