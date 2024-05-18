###                     PRML ASSIGNMENT - 3 (PART - 1)                  ###

## importing necessary libraries
import numpy as np
import pandas as pd
import random
import os
import sys
from sklearn.model_selection import train_test_split
from numpy import random
random.seed(0)  ## setting the seed to 0

random_float = random.uniform(-1, 1)
# print(random_float)
# print(random.randint(0,20))   ## printing the random numbers

## taking initial weights between -1 to 1
w0 = random.uniform(-1, 1)
w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
w3 = random.uniform(-1, 1)
w4 = random.uniform(-1, 1)

## function to generate the random points
def func(x1, x2, x3, x4):
  res = w0 + w1 * x1 + w2 * x2 + w3 * x3
  return res

## dictionary to store the dataset
dataset = {
    "x1" : [],
    "x2" : [],
    "x3" : [],
    "x4" : [],
    "label" : []
}

## generating the random points and storing in dictionary
for i in range(0,5000):
  x1 = random.randint(0,100);
  x2 = random.randint(0,100);
  x3 = random.randint(0,100);
  x4 = random.randint(0,100);

  result = func(x1, x2, x3, x4)
  output = None
  if(result >= 0):
   output = 1
  else:
    output = 0

  dataset["x1"].append(x1)
  dataset["x2"].append(x2)
  dataset["x3"].append(x3)
  dataset["x4"].append(x4)
  dataset["label"].append(output)


df = pd.DataFrame(dataset)

# df.to_csv('B22CS003_data.txt', sep=' ', index=False) ## saving the dataset as .txt file
with open('B22CS003_data.txt', 'w') as f:
    # Loop through each row in the DataFrame
    f.write("x1 x2 x3 x4 label" + "\n")
    for index, row in df.iterrows():
        # Convert each row to a string with space-separated values
        line = ' '.join(map(str, row))
        # Write the line to the file
        f.write(line + '\n')

# print(df["label"].value_counts())  ## printing the vaule count of label to know about class distribution

## splitting the dataset into train and test
X_data = df.values
Y_data = df["label"].values

datax_train, datax_test, datay_train, datay_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
datax_test = datax_test[:, :-1]
df1 = pd.DataFrame(datax_train, columns=["x1","x2", "x3", "x4", "label"])
df2 = pd.DataFrame(datax_test, columns=["x1","x2", "x3", "x4"])

## creating the train.txt file
with open('B22CS003_train.txt', 'w') as f:
    # Loop through each row in the DataFrame
    f.write("x1 x2 x3 x4 label" + "\n")
    for index, row in df1.iterrows():
        # Convert each row to a string with space-separated values
        line = ' '.join(map(str, row))
        # Write the line to the file
        f.write(line + '\n')


## creating the test.txt file
with open('B22CS003_test.txt', 'w') as f:
    # Loop through each row in the DataFrame
    f.write("x1 x2 x3 x4" + "\n")
    for index, row in df2.iterrows():
        # Convert each row to a string with space-separated values
        line = ' '.join(map(str, row))
        # Write the line to the file
        f.write(line + '\n')

filename = input("Enter train.txt path : ")

## checking if the file exits or not
if os.path.exists(filename) == False:
    print(f"The file {filename} does not exist.")
    sys.exit()

# Read the txt file
with open(filename, 'r') as file:
    lines = file.readlines()

# Extract size of input
size = int(lines[0])

# Create a list to store data
data = []

# Extract rows of dataset
for line in lines[1:]:
    row = list(map(int, line.strip().split()))
    data.append(row)

# Create DataFrame
columns = [f'x{i}' for i in range(1, size + 1)] + ['label']
df_train = pd.DataFrame(data, columns=columns)

# print(df_train)

## importing MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X = df_train.drop(columns=["label"]).values  ## converting to array
y = df_train["label"].values  ## converting to array
# print(X)
## performing the transformation 
X = scaler.fit_transform(X)


X_train = X
y_train = y

## performing train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

## function to calculate the generate label
def generate_label(x1,x2, x3, x4):
   result = func(x1, x2, x3, x4)
   output = None
   if(result >= 0):
    output = 1
   else:
    output = 0

   return output
 
## writing the step function
def step(z):
    return 1 if z>0 else 0

## perceptron learning algorithm
def perceptron(X,y):
    X = np.insert(X,0,1,axis=1)  ## adding the bias to the array
    weights = np.ones(X.shape[1])  ## taking the random weights
    lr = 0.1
    ## iterating with epochs
    for i in range(1000):
        j = np.random.randint(0,X.shape[0])
        y_hat = step(np.dot(X[j],weights))  ## predicting the value
        weights = weights + lr*(y[j]-y_hat)*X[j]  ## updating the weights

    return weights

weights = perceptron(X,y)

## creating the weights.txt file
with open('B22CS003_weights.txt', 'w') as f:
    arr = weights.tolist()
    for i in arr:
       f.write(str(i) + '\n')
       
print("Weights calculated and saved to B22CS003_weights.txt")  
print(weights)

