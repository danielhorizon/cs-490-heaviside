# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:03:04 2020

https://link.springer.com/chapter/10.1007/978-3-030-52249-0_27#enumeration


@author: rlaug
"""

#Imports:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn import cross_validation
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Polygon
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.regularizers import l1
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier 

###############################################################################

#Read in file from UCI Repository:
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
                  delimiter = ';')

# I'm assuming you can work with the data-frame according 
# to your own methods, recall I performed (1) z-score standardization,
# ((x - mean) / sd), (2) outlier removal (> 3SD +/- removed),
# (3) PCA to extract max. variance (11 PC's), not dimensionality
# reduction, and (4) class reduction, all described in detail in the paper.

# Feel free to use your own techniques here, upstream of MNN model.

# MNN with ADAM optimizer: ####################################################

# Note: Wine data set has 11 features, transformed via PCA to 11 PC's
# Note2: Reduced class size from 7 to 4, as features 
# do not have enough predictive power to disciminate 7 classes.

# For MLP, the input vectors are all on same scale (z-score standardization), 
# and converted to numpy arrays, obviously train/test split (80/20 ratio),
# only train and test, no separate validation holdout set was used since
# the data set is so small (4,898 rows by 11 feature columns), I just 
# couldn't afford the luxury of a separate validation hold out set.

model = Sequential()
model.add(Dense(1000, activation='relu', 
               kernel_initializer='random_normal', input_dim=11)) # alcohol%, pH, etc.
model.add(Dense(500, activation='relu', 
               kernel_initializer='random_normal'))
model.add(Dense(250, activation='relu', 
               kernel_initializer='random_normal'))
model.add(Dense(4, activation='softmax',   # reduced class size from 7 to 4, see paper
               kernel_initializer='random_normal', W_regularizer=l2(0.1))) # high L2!

model.compile(optimizer ='adam',loss='categorical_crossentropy',  # used ADAM
                   metrics =['accuracy'])  #ADAM optimizer performed higher
model.summary()

#Fit model:
history = model.fit(Xo_train,yo_train, batch_size=512, # fairly large batch size
          epochs=75, validation_data =(Xo_test, yo_test))

#Plot graph of Loss vs Categorical cross entropy
fig = plt.figure(figsize=(6,4))

#Summarize history for loss:
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], 'g--')
plt.title('Neural Network Model Loss')
plt.ylabel('Categorical Crossentropy (CCE)')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Testing Loss'], loc='upper right')
print ("Loss after final iteration: ", history.history['val_loss'][-1])
plt.show()

#Define function:
def ohe_to_classes(y):
    '''
        converts one hot encoding to classes
        y: a list of one-hot-encoded classes of data points
    '''
    return [np.argmax(v) for v in y]

#Predicted classes:
predicted_all = model.predict_classes(np.array(Xo_test))
print("predicted classes: {}".format(predicted_all))

#Convert one-hot-encoding to actual classes: 0 - 3
y_true_classes = ohe_to_classes(yo_test)

#Print metrics:
print('accuracy', accuracy_score(predicted_all, y_true_classes))
confusion_mat = confusion_matrix(predicted_all, y_true_classes)
print("confusion matrix\n{}\n\n".format(confusion_mat))
print(classification_report(predicted_all, y_true_classes))
print('Accuracy:', round(accuracy_score(predicted_all, y_true_classes),2))

#Extra visualizations:
conf_mx = confusion_mat
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

###############################################################################

# Show pretty confusion matrix:
fig, ax = plt.subplots()
min_val, max_val = 0, 4
conf_mx = confusion_mat
ax.matshow(conf_mx, cmap=plt.cm.Blues)

for i in range(4):
    for j in range(4):
        c = conf_mx[j, i]
        ax.text(i,j, str(c), va='center', ha='center')
        
###############################################################################

#Processing:
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

#Fill the diagonal with zeros to keep only the errors, 
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

#End multilayer neural network code block #####################################








