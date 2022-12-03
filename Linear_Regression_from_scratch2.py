import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 

class LinearRegression():

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations 

    def fit(self, X,Y): 
        self.m, self.n = X.shape 
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X

    # Gradient Descent 
        for i in range(self.iterations):
            self.update_weights()
        return self
    
    # Update Weight Helper Function 
    def update_weights(self): 
        Y_pred = self.predict(self.X)
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) // self.m 
        db = -2 *np.sum(self.Y - Y_pred)/self.m

        self.W = self.W - self.learning_rate * dW 
        self.b = self.b - self.learning_rate * db 
        return self 

    def predict(self, X):
        return X.dot(self.W) + self.b 

# Tutorial Source
# https://www.geeksforgeeks.org/linear-regression-implementation-from-scratch-using-python/
