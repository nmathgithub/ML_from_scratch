# Import necessary modules 
import numpy as np 
from scipy.spatial.distance import cdist 
from sklearn.datasets import load_digits 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 


# Step 1: Define main function 
def get_kmeans(x, k, num_iterations): 
    # Randomly initialize centroids 
    idx = np.random.choice(len(x), k, replace=False) 
    centroids = x[idx, :] 

    # Find distance between Centroids and all data points
    distances = cdist(x, centroids, 'euclidean')

    # Align with the centroid with the minimal distance
    points = np.array([np.argmin(i) for i in distances])

    # Repeat with number of iterations 
    for i in range(num_iterations): 
        centroids = []
        for j in range(k): 
            temp_centroid = x[points==j].mean(axis=0)
            centroids.append(temp_centroid)
        centroids = np.vstack(centroids) #Update Centroids

        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])

    return points 

# Load Data for Testing 
data = load_digits().data 
pca = PCA(2) 

# Transform the data 
df = pca.fit_transform(data)

# Apply function 
label = get_kmeans(df, 10, 100)

# Visualize 
u_labels = np.unique(label)
for i in u_labels: 
    plt.scatter(df[label == i, 0], df[label==i, 1], label = i) 
plt.legend()
plt.show()

# Tutorial Source:
# https://www.askpython.com/python/examples/k-means-clustering-from-scratch