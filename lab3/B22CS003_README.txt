### README FILE ###

*****  train.py *********
## It has the code to generate the synthetic dataset of 5000 points.
## It is generating dataset and saving to "B22CS003_data.txt"
## Performing split to store the "B22CS003_train.txt" and "B22CS003_test.txt"
## training model and storing weights in "B22CS003_weights.txt"

*****  test.py **********
## It is reading the "test.txt" file 
## Making prediction and then converting prediction to "labels.csv"



*************** HOW TO RUN MY CODE ******************

## Write all the things in the command line
## Step 1  : cd B22CS003   // if not in my folder
## Step 2  : python B22CS003_test.py
## Step 3  : Enter "train.txt" file in the format given in assignment
## Step 4  : Enter "test.text" file in the format given in assignment

*************** DEMO RUN ****************************
python B22CS003_test.py
Enter train.txt path : train.txt
Weights calculated and saved to B22CS003_weights.txt
[-0.5  -0.1   0.1   0.82  0.5 ]
Enter test.txt path : test.txt
*************** PREDICTION ******************
[1 1 1]

******************** HOW TO PRINT ACCURACY *************
## run generate_label() function to generate label of test