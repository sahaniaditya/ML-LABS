##  Complete this code by writing the function definations
##  Compute following terms and print them:
#1. Difference of class wise means = m1 - m2
#2. Total Within-class Scatter Matrix S_W
#3. Between-class Scatter Matrix S_B
#4. The EigenVectors of matrix S_W-1SB corresponding to highest EigenValue
#5. For any input 2-D point, print its projection according to LDA.

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
## importing necessary libraries

def ComputeMeanDiff(X):
  ## extracting the X part and y part
  X_train = X[:, :-1]
  y_train = X[:, -1]
  ## storing number of classes
  classes = np.unique(y_train)
  label_classes = np.sort(y_train)
  MEAN = []
  ## iterating through class
  for cls in classes:
    X_cls = X_train[y_train == cls]  ## extracting the matrix
    mean = np.zeros((1, X.shape[1]))
    mean = np.mean(X_cls, axis = 0)
    MEAN.append(mean)  ## appending in MEAN list

  return (MEAN[0] - MEAN[1])  ## returning the list



def ComputeSW(X):
  ## extracting the X part of features and y label part
  X_train = X[:, :-1]
  y_train = X[:, -1]

  ## storing number of classes for iterationg
  classes = np.unique(y_train)
  label_classes = np.sort(y_train)
  ## initializing the scatter_within matrix with 0.
  S_W = np.zeros((X_train.shape[1], X_train.shape[1]))

  ## iterating through classes
  for cls in classes:
      ## extracting rows of class c
      X_c = X_train[y_train == cls]
      ## calculating the mean
      scatter_within_vec = np.ones((2,2))
      mean_vec = np.mean(X_c, axis=0)
      ## initializing with zeroes
      S_c = np.zeros((X_train.shape[1], X_train.shape[1]))
      ## iterating for each class sample
      for sample in X_c:
          ## calculating the matrix
          deviation = (sample - mean_vec).reshape(-1, 1)
          scatter_within_range = np.arange(0,X.shape[0])
          S_c += deviation.dot(deviation.T)

      S_W += S_c

  return S_W  ## returning the matrix

def ComputeSB(X):
    ## computing the overall mean
    mean = np.mean(X[:,:-1], axis=0)

    ## storing the number of classes
    classes = np.unique(X[:, -1])
    label_classes = np.sort(classes)
    ## initializing the scatter_betwen matrix with zeroes
    S_B = np.zeros((X.shape[1] - 1, X.shape[1] - 1))

    ## iterating through each class
    for cls in classes:

        X_c = X[X[:, -1] == cls][:, :-1]
        scatter_between_vec = np.ones((2,2))
        mean_vec = np.mean(X_c, axis=0)
        ## calculating the deviation from normal mean
        deviation = mean_vec - mean
        scatter_between_range = np.arange(0,X.shape[0])
        deviation = deviation.reshape(-1, 1)
        
        S_B += X_c.shape[0] * deviation.dot(deviation.T) ## appending sum in final matrix
    
    mean_diff = ComputeMeanDiff(X) 
    mean_diff = mean_diff.reshape(1,2)
    SB = mean_diff.T.dot(mean_diff)   
    return SB  ## returning the matrix

def GetLDAProjectionVector(X):
  ## calculating the S_W-1S_B dot product
  S_W = ComputeSW(X)
  S_B = ComputeSB(X)
  A = np.linalg.inv(S_W).dot(S_B)  ## using the linearalgebra 
  ## calculating the eignevalues and eigenvectors
  eigenvalues, eigenvectors = np.linalg.eig(A)
  eigenvectors  = eigenvectors.T
  ## sorting the eigenvalues and eigenvectors
  indices = np.argsort(abs(eigenvalues))[::-1]
  eigenvalues = eigenvalues[indices]
  eigenvectors = eigenvectors[indices]
  return eigenvectors[0]  ## returning the projection vector

def project(x,y,w):
  ## converting point to array
  array = np.array([x,y])
  return np.dot(array, w.T)

column_names = ["x1", "x2", "label"]


df = pd.read_csv("data.csv", names=column_names)

X = np.empty((0, 3))
with open('data.csv', mode ='r')as file:
  csvFile = csv.reader(file)
  for sample in csvFile:
        X = np.vstack((X, sample))

X = X.astype(float)
print(X)
print("Shape : ", X.shape)
print("\n")

# X Contains m samples each of formate (x,y) and class label 0.0 or 1.0
opt=int(input("Input your option (1-5): "))

match opt:
  case 1:
    meanDiff=ComputeMeanDiff(X)
    print(meanDiff)
  case 2:
    SW=ComputeSW(X)
    print(SW)
  case 3:
    SB=ComputeSB(X)
    print(SB)
  case 4:
    w=GetLDAProjectionVector(X)
    print(w)
  case 5:
    x=int(input("Input x dimension of a 2-dimensional point :"))
    y=int(input("Input y dimension of a 2-dimensional point:"))
    w=GetLDAProjectionVector(X)
    print(project(x,y,w))