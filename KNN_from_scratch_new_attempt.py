import numpy as np
from scipy.stats import mode 

# Step 1: Calculate Euclidean distance 
def get_euclidean_distance(x1, x2): 
    distances_new = np.sqrt(np.sum((x1-x2) **2))
    return distances_new

# Step 2: Calculate KNN 
def get_KNN(x_train, y, x_test, k):
    labels_new = []

    # a) Loop through data points that need to be classified 
    for i in x_test: 
        # Create New array to store distances 
        point_dist = [] 
        # Loop through each training data 
        for j in range(len(x_train)):
            distances = get_euclidean_distance(np.array(x_train[j,:]), i)
            point_dist.append(distances) 
        point_dist = np.array(point_dist)
        # b) Sort array but preserve index 
        new_dist = np.argsort(point_dist)[: k] 
        # c) Get Labels of K datapoints 
        labels = y[new_dist] 
        # d) Majority Voting 
        lab = mode(labels, keepdims = True) 
        lab = lab.mode[0] 
        labels_new.append(lab)
    return labels_new 

# Load Data for Testing 
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris 
from numpy.random import randint

iris = load_iris()
X = iris.data 
y = iris.target 

train_idx = xxx = randint(0,150,100)
X_train = X[train_idx] 
y_train = y[train_idx]

test_idx = xxx = randint(0, 150,50)
X_test = X[test_idx]
y_test = y[test_idx] 

y_pred = get_KNN(X_train, y_train, X_test, 7)
accuracy_score(y_test, y_pred)
# print(accuracy_score)

# Tutorial Source
# https://www.askpython.com/python/examples/k-nearest-neighbors-from-scratch


