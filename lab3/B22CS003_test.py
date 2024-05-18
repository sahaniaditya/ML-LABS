## importing necessary libraries
import numpy as np
import os
import sys
import csv
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from numpy import random
from sklearn.linear_model import Perceptron
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import B22CS003_train as train

# Read the txt file
test_file_path = input("Enter test.txt path : ")

## checking if the file exits or not
if os.path.exists(test_file_path) == False:
    print(f"The file {test_file_path} does not exist.")
    sys.exit()


## reading the test.txt and converting to array
data_test = pd.read_csv(test_file_path, sep = ' ', skiprows=1, header = None) ## converting the text file to csv file
X_test = data_test.iloc[:, :].values  ## converting to 2D array


## generating the label
labels = []
for i in range(0, X_test.shape[0]):
   x1 = X_test[i][0]
   x2 = X_test[i][1]
   x3 = X_test[i][2] 
   x4 = X_test[i][3]
   label = train.generate_label(x1, x2, x3, x4)
   labels.append(label)


label_array = np.array(labels)

# print(label_array)


# print("*************** X_TEST ******************")
# print(X_test)

X_test = train.scaler.transform(X_test)  ## transforming the X_test



""" PERCETPRON USING SKLEARN """

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(train.X_train, train.y_train)
y_pred = clf.predict(X_test)
# accuracy_sklearn = accuracy_score(train.y_test, y_pred)
# print("Accuracy using Perceptron from Sklearn : ", accuracy_sklearn)


""" PERCEPTRON FROM SCRATCH """

## writing the predict function
def predict(X):
  prediction = []
  X = np.insert(X,0,1,axis=1)
  for j in range(X.shape[0]):
   y_hat = train.step(np.dot(X[j],train.weights))
   prediction.append(y_hat)
  return prediction

""" Commenting the accuracy as it is not needed """
pred_array = np.array(predict(X_test))  ## array to store the prediction
# accuracy_scratch = accuracy_score(train.y_test, pred_array)
# print("Accuracy using Perceptron Training Algorithm : ", accuracy_scratch)
print("*************** PREDICTION ******************")
print(pred_array)

# Convert the array to a comma-separated file
with open('labels.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(pred_array)

    """  Uncomment the code to print accuracy """

# accuracy_scratch = accuracy_score(label_array, pred_array)    
# print("Accuracy using Perceptron Training Algorithm : ", accuracy_scratch * 100)