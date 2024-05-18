"""# PART 2

## Task-1 (Data Preprocessing)
"""

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from numpy import random
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## importing the necessary libraries

## loading the dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

## extracting the data and target values from teh lfw_people
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]


## printing about the datasets
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
print("Target Names : ", target_names)

## we have 1288 images each of height = 50 and width = 37 (Total number of pixels : 50 x 37 = 1850)

## splitting the data in the ratio 80 : 20 using train_test_split
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(X_train.shape)

"""## Task-2 (EigenFaces Implementation)

### PCA from Scratch
"""

## PCA implementation using EigenVectors and EigenValues

class PCA_Scratch:

  ## defining the constructor of PCA class
  def __init__(self, n_components):
    self.mean_ = None
    self.std_ = None
    self.n_components = n_components
    self.components_ = None
    self.principal_components = None
    self.explained_variance_ratio_ = None

  def  fit(self, X):
    mean_array = np.mean(X, axis=0)  ## calculating the mean for each feature
    self.mean_ = mean_array
    std_array = np.std(X, axis=0)  ## calculating the std for each feature
    self.std_ = std_array
    X = X - mean_array  ## mean centering the feature array
    cov_matrix = np.dot(X.T, X) / (X.shape[0])  ## getting the covariance matrix

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  ## getting the eigenvalus and eigenvectors

    sorted_indices = np.argsort(eigenvalues)[::-1]  ## selecting the maximum eigen value indices
    sorted_eigenvalues = eigenvalues[sorted_indices]
    # Calculate explained variance ratio
    total_variance = np.sum(sorted_eigenvalues)
    self.explained_variance_ratio_ = sorted_eigenvalues[0 : self.n_components] / total_variance

    # print(eigenvalues)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]  ## sorting the eigen vectors corresponding to their eigen values
    self.components_ = sorted_eigenvectors
    self.principal_components = sorted_eigenvectors[:, :self.n_components]


  def transform(self, X):
     ## transforming the dataset
     transformed_data = np.dot(X, self.principal_components)
     return transformed_data

"""# Task 3 and Task 4 (Model Training and Evaluation)"""

## projecting data and then reconstructing it for n_components_ = 2

mean = scaler.mean_
std = scaler.scale_
components = 100
pca = PCA_Scratch(n_components = components)  ## using PCA from scratch
pca.fit(X)
projected_data = pca.transform(X)  ## projected data

reconstructed_data = (np.dot(projected_data, pca.principal_components.T) * std) + mean  ## reconstructing the data


# Plot the original and reconstructed data
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], label='Original Data')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], label='Reconstructed Data', color='orange')
plt.title('Reconstructed Data (PCA)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.suptitle('PCA from Scratch')
plt.tight_layout()
plt.show()

## Plotting the dataset for different classes
pca1 = PCA_Scratch(n_components = 2)
pca1.fit(X)
pca1_data = pca1.transform(X)
pca1_df = pd.DataFrame(pca1_data, columns=["x1", "x2"])
pca1_dff = pd.DataFrame(y, columns=["target"])
new_df1 = pd.concat([pca1_df, pca1_dff], axis=1)
plt.figure(figsize=(10, 5))
sns.scatterplot(x="x1", y="x2", hue="target", data=new_df1, palette="viridis")
plt.title("Data Visualization Using PCA_Scratch")
plt.show()

## Implementing the eigenfaces using PCA_Scratch class
lfw_dataset = fetch_lfw_people(min_faces_per_person=10, resize=0.4)  ## taking min_faces_per_person = 10

# Extract face images and target names
faces = lfw_dataset.images
n_samples, h, w = faces.shape
print("Images Dataset Size : ",n_samples, h, w)
target_names = lfw_dataset.target_names

# Flatten the images into 1D vectors
X1 = faces.reshape(n_samples, -1)
print("Flatten Image Shape : ",X1.shape)
sc = StandardScaler()
X1 = sc.fit_transform(X1)


# Choose faces randomly for display and project later
num_faces = n_samples
chosen_indices = np.random.choice(n_samples, num_faces, replace=False)
chosen_faces = X1[chosen_indices]

# Compute PCA: we are using in-built function from Sklearn
n_components = 1850  # Number of principal components (eigenfaces)
pca = PCA_Scratch(n_components=n_components)
pca.fit(X1)

# Extract eigenfaces
compo = pca.principal_components.T

# eigenfaces1 = compo.reshape((n_components, h, w))
eigenfaces = compo.reshape((n_components, h, w))
# print_eigenfaces1 = eigenfaces

# Plot the top-10 eigenfaces
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow((eigenfaces[i]), cmap='gray')
    # print(f"Eigen {i } : ", eigenfaces[i])
    plt.title(f"Eigenface {i + 1}")
    plt.xticks(())
    plt.yticks(())

plt.suptitle("Top-10 Eigenfaces Using PCA_Scratch")
plt.show()

# Choose a face to reconstruct (index within the chosen faces)
face_to_reconstruct_index = 0

# Specify number of eigenfaces to use for reconstruction
k = 1850

# Project the chosen face onto the k eigenfaces
chosen_face = chosen_faces[face_to_reconstruct_index]  ## chosing a face
chosen_face_flattened = chosen_face.reshape(1, -1)   ## flattening the face
eigenfaces_to_use = eigenfaces[:k]
projected_face = np.dot(chosen_face_flattened - pca.mean_, eigenfaces_to_use.reshape(k, -1).T)  ## projecting it using eigen vectors


# Reconstruct the face using the k eigenfaces
reconstructed_face = pca.mean_ + np.dot(projected_face, eigenfaces_to_use.reshape(k, -1))  ## reconstructing the data

# Reshape the reconstructed face to its original shape
reconstructed_face = reconstructed_face.reshape(h, w)
plt.subplots_adjust(hspace=0.5)
# Plot the original and reconstructed faces
plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.imshow(chosen_face.reshape(h, w), cmap='gray')  # Reshape to original shape before plotting
plt.title('Original Face')
plt.xticks(())
plt.yticks(())

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_face, cmap='gray')
plt.title(f'Reconstructed Face using {k} Eigenfaces')
plt.xticks(())
plt.yticks(())

plt.tight_layout()
plt.show()

## mean face
plt.imshow(mean.reshape(h,w), cmap='gray')
plt.title("Mean Face")
plt.show()

# Plot the explained variance ratio
ratio = pca.explained_variance_ratio_
cumulative_sum = np.cumsum(ratio)  ## takign cumulative sum

sns.lineplot(x=np.arange(1,ratio.shape[0] + 1),y=cumulative_sum)  ## line plotting the explained_variance_
sns.set_style("whitegrid")
sns.set_context("notebook")
plt.title("Variance Using PCA_Scratch")
plt.show()

"""### PCA from Sklearn"""

mean = scaler.mean_
std = scaler.scale_
components = 100
pca = PCA(n_components = components) ## using PCA from sklearn
pca.fit(X)
projected_data = pca.transform(X)  ## projected data

reconstructed_data = (np.dot(projected_data, pca.components_) * std) + mean  ## reconstructing the data


# Plot the original and reconstructed data
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], label='Original Data')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], label='Reconstructed Data', color='orange')
plt.title('Reconstructed Data (PCA)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.suptitle('PCA from Sklearn')
plt.tight_layout()
plt.show()

## plotting dataset using PCA from sklearn
pca1 = PCA(n_components = 2)
pca1.fit(X)
pca1_data = pca1.transform(X)
pca1_df = pd.DataFrame(pca1_data, columns=["x1", "x2"])
pca1_dff = pd.DataFrame(y, columns=["target"])
new_df1 = pd.concat([pca1_df, pca1_dff], axis=1)
plt.figure(figsize=(10, 5))
sns.scatterplot(x="x1", y="x2", hue="target", data=new_df1, palette="viridis")
plt.title("Data Visualization using Sklearn")

## Implementign the eigenfaces  using PCA from Sklearn

from sklearn.decomposition import PCA
lfw_dataset = fetch_lfw_people(min_faces_per_person=10, resize=0.4)

# Extract face images and target names
faces = lfw_dataset.images
n_samples, h, w = faces.shape
print("Images Dataset Size : ",n_samples, h, w)
target_names = lfw_dataset.target_names

# Flatten the images into 1D vectors
X2 = faces.reshape(n_samples, -1)
print("Flatten Image Shape : ",X2.shape)


# Choose faces randomly for display and project later
num_faces = n_samples
chosen_indices = np.random.choice(n_samples, num_faces, replace=False)
chosen_faces = X2[chosen_indices]

# Compute PCA: we are using in-built function from Sklearn
n_components = 1850  # Number of principal components (eigenfaces)
pca = PCA(n_components=n_components)
pca.fit(X2)

# Extract eigenfaces

eigenfaces = pca.components_.reshape((n_components, h, w))
# print_eigenfaces2 = eigenfaces

# Plot the top-10 eigenfaces
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(eigenfaces[i], cmap='gray')
    # print(f"Eigen {i } : ", eigenfaces[i])
    plt.title(f"Eigenface {i + 1}")
    plt.xticks(())
    plt.yticks(())

plt.suptitle("Top-10 Eigenfaces Using Sklearn")
plt.show()

# Choose a face to reconstruct (index within the chosen faces)
face_to_reconstruct_index = 0

# Specify number of eigenfaces to use for reconstruction
k = 1850

# Project the chosen face onto the k eigenfaces
chosen_face = chosen_faces[face_to_reconstruct_index]
chosen_face_flattened = chosen_face.reshape(1, -1)
eigenfaces_to_use = eigenfaces[:k]
projected_face = np.dot(chosen_face_flattened - pca.mean_, eigenfaces_to_use.reshape(k, -1).T)

# Reconstruct the face using the k eigenfaces
reconstructed_face = pca.mean_ + np.dot(projected_face, eigenfaces_to_use.reshape(k, -1))

# Reshape the reconstructed face to its original shape
reconstructed_face = reconstructed_face.reshape(h, w)

plt.subplots_adjust(hspace=0.5)
# Plot the original and reconstructed faces
plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.imshow(chosen_face.reshape(h, w), cmap='gray')  # Reshape to original shape before plotting
plt.title('Original Face')
plt.xticks(())
plt.yticks(())

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_face, cmap='gray')
plt.title(f'Reconstructed Face using {k} Eigenfaces')
plt.xticks(())
plt.yticks(())

plt.tight_layout()
plt.show()

# Plot the explained variance ratio
ratio = pca.explained_variance_ratio_
cumulative_sum = np.cumsum(ratio)
sns.lineplot(x=np.arange(1,ratio.shape[0] + 1),y=cumulative_sum)
sns.set_style("whitegrid")
sns.set_context("notebook")
plt.title("Variance using Sklearn")
plt.show()

"""# Task-5 (Experimentation with different n_components)"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

"""### Using PCA from Scratch"""

dic = {
    "n_components" : [],
    "KNN_train" : [],
    "KNN_test" : [],
    "DTC_train" : [],
    "DTC_test" : [],
    "RFC_train" : [],
    "RFC_test" : []
}
for i in range(2,500, 10):
    pca = PCA_Scratch(n_components=i)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    dic["n_components"].append(i)

    # K-Nearest Neighbors (KNN)
    knn = KNeighborsClassifier(n_neighbors=60)
    knn.fit(X_train_pca, y_train)
    knn_pred = knn.predict(X_test_pca)
    knn_pred1 = knn.predict(X_train_pca)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_accuracy1 = accuracy_score(y_train, knn_pred1)
    dic["KNN_test"].append(knn_accuracy)
    dic["KNN_train"].append(knn_accuracy1)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=10)
    dt.fit(X_train_pca, y_train)
    dtc_pred = dt.predict(X_test_pca)
    dtc_pred1 = dt.predict(X_train_pca)
    dtc_accuracy = accuracy_score(y_test, dtc_pred)
    dtc_accuracy1 = accuracy_score(y_train, dtc_pred1)
    dic["DTC_test"].append(dtc_accuracy)
    dic["DTC_train"].append(dtc_accuracy1)


    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=100, max_depth=10)
    rf.fit(X_train_pca, y_train)
    rf_pred = rf.predict(X_test_pca)
    rf_pred1 = rf.predict(X_train_pca)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_accuracy1 =  accuracy_score(y_train, rf_pred1)
    dic["RFC_test"].append(rf_accuracy)
    dic["RFC_train"].append(rf_accuracy1)

"""## Prediction"""

## final prediction using KNN and plotting the faces
knn = KNeighborsClassifier(n_neighbors=60)
knn.fit(X_train_pca, y_train)
knn_pred = knn.predict(X_test_pca)
knn_accuracy = accuracy_score(y_test, knn_pred)

array = random.randint(0, 100, 10)
plt.figure(figsize=(10, 5))
j = 0
for i in array:
  plt.subplot(2, 5, j + 1)
  img = X_test[i].reshape(h, w)
  plt.imshow(img , cmap='gray')
  plt.title(f"Predicted: {target_names[knn_pred[i]]} \n Actual: {target_names[y_test[i]]}", fontsize=8)
  plt.xticks(())
  plt.yticks(())
  j = j + 1

plt.show()

# from sklearn.metrics import confusion_matrix
# true_labels = y_test
# predicted_labels = knn_pred
# # Calculate confusion matrix
# cm = confusion_matrix(true_labels, predicted_labels)
# # Plot confusion matrix as heatmap
# plt.figure(figsize=(10, 6))
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['Class 0', 'Class 1', "Class 2", "Class 3", "Class 4","Class 5", "Class 6"], yticklabels=['Class 0', 'Class 1', "Class 2", "Class 3", "Class 4","Class 5", "Class 6"])
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.title('Confusion Matrix')
# plt.show()

from sklearn.metrics import classification_report
true_labels = y_test
predicted_labels = knn_pred

report = classification_report(true_labels, predicted_labels, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'])
print(report)

df1 = pd.DataFrame(dic, columns=["n_components","KNN_train","KNN_test", "DTC_train", "DTC_test" ,"RFC_train", "RFC_test"])

df1.shape

fig, axs = plt.subplots(1, 3, figsize=(15, 6))

sns.lineplot(data=df1[["KNN_train", "KNN_test"]], ax=axs[0]) 
sns.lineplot(data=df1[["DTC_train", "DTC_test"]], ax=axs[1])  
sns.lineplot(data=df1[["RFC_train", "RFC_test"]], ax=axs[2])  

axs[0].set_title('KNN')
axs[1].set_title('DTC')
axs[2].set_title('RFC')

plt.show()

"""### Using PCA from sklearn"""

dic = {
    "n_components" : [],
    "KNN_train" : [],
    "KNN_test" : [],
    "DTC_train" : [],
    "DTC_test" : [],
    "RFC_train" : [],
    "RFC_test" : []
}
for i in range(2,500, 10):
    pca = PCA(n_components=i)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    dic["n_components"].append(i)

    # K-Nearest Neighbors (KNN)
    knn = KNeighborsClassifier(n_neighbors=60)
    knn.fit(X_train_pca, y_train)
    knn_pred = knn.predict(X_test_pca)
    knn_pred1 = knn.predict(X_train_pca)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_accuracy1 = accuracy_score(y_train, knn_pred1)
    dic["KNN_test"].append(knn_accuracy)
    dic["KNN_train"].append(knn_accuracy1)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=10)
    dt.fit(X_train_pca, y_train)
    dtc_pred = dt.predict(X_test_pca)
    dtc_pred1 = dt.predict(X_train_pca)
    dtc_accuracy = accuracy_score(y_test, dtc_pred)
    dtc_accuracy1 = accuracy_score(y_train, dtc_pred1)
    dic["DTC_test"].append(dtc_accuracy)
    dic["DTC_train"].append(dtc_accuracy1)


    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=100, max_depth=10)
    rf.fit(X_train_pca, y_train)
    rf_pred = rf.predict(X_test_pca)
    rf_pred1 = rf.predict(X_train_pca)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_accuracy1 =  accuracy_score(y_train, rf_pred1)
    dic["RFC_test"].append(rf_accuracy)
    dic["RFC_train"].append(rf_accuracy1)

df2 = pd.DataFrame(dic, columns=["n_components","KNN_train","KNN_test", "DTC_train", "DTC_test" ,"RFC_train", "RFC_test"])

df2.head()

fig, axs = plt.subplots(1, 3, figsize=(15, 6))

sns.lineplot(data=df2[["KNN_train", "KNN_test"]], ax=axs[0])  

sns.lineplot(data=df2[["DTC_train", "DTC_test"]], ax=axs[1])  

sns.lineplot(data=df2[["RFC_train", "RFC_test"]], ax=axs[2])  

axs[0].set_title('KNN')
axs[1].set_title('DTC')
axs[2].set_title('RFC')

plt.show()