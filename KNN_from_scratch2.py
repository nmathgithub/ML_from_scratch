import numpy as np
from collections import Counter  

# Step I 
# We first calculate Euclidean distance 

def euc_distance(x1, x2):
    distance = 0 
    for i in range(len(x1)-1):
        distance += (x1[i]-x2[i])**2
    return np.sqrt(distance)

def euc_distance2(x1, x2):
    distance = np.sqrt(np.sum(np.subtract(x1,x2)**2))
    return distance 
# Test Step 1 
dataset = [
    [2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]
          ]
test = [8.675418651, 2.088626775,1]
for i in dataset: 
    distance = euc_distance(test, i)
    distance2 = euc_distance2(test, i)
    # print(distance, distance2)

# Step 2: Find the Nearest Neighbors 
def get_KNN (train, test, k): 
    distance = list() 
    data = []
    for i in train: 
        dist = euc_distance2(test, i)
        distance.append(dist)
        data.append(i)
    # print(distance)
    print(data)
    distance = np.array(distance) 
    data = np.array(data)
    # print(distance)
    # print(data)
    index_dist = distance.argsort()
    data = data[index_dist]
    # print(data)
    neighbors = data[:k]
    return neighbors 

# Test Method 
# print(get_KNN(dataset, test, 5))

# Predict the Output 
def KNN_classifier(train, test, k): 
    neighbors = get_KNN(train, test, k)
    Classes = []
    for i in neighbors: 
        Classes.append(i[-1])
    prediction = max(Classes, key = Classes.count)
    return prediction 

# Test Method 
print(KNN_classifier(dataset, dataset[0], 4))

# *Note: Tutorial Resource from 
# https://medium.com/@ojaswini51/k-nearest-neighbors-in-python-from-scratch-3611c2d7517e#:~:text=k-Nearest-Neighbors%20in%20Python%20%28from%20scratch%29%201%20Step%201%3A,%3A%20Classification%20predicted%20from%204%20most%20similar%20neighbors.