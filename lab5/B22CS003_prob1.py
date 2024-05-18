# PART - 1(K-Means Clustering)

### Task - A


## importing necessary libraries
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

## importing the image
img = cv2.imread("/content/test.png")
rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height, width, channels = rgb_image.shape
## extracting the pixels from the image
rgb_data = []

for y in range(height):
  for x in range(width):

      r, g, b = rgb_image[y, x]
      rgb_data.append([r, g, b])

len(rgb_data)  ## total no. of pixels

rgb_array = np.array(rgb_data)  ## converting to array

## function to compute the centroid of the data
def computeCentroid(X):
  mean = np.mean(X, axis=0)
  return mean

computeCentroid(rgb_array)  ## printing the centroid of the original data

### Task - B

## function to group the clusters
def group_clusters(X, centroids):
  dis = []
  dis = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  ## calculating the distances
  groups = np.argmin(dis, axis=1)
  return groups

## functions to assign the new centroids
def new_centroids(X,cluster_group):
    new_cens = []
    cluster = np.unique(cluster_group)  ## extracting the unique clusters
    for type in cluster:
        new_cens.append(X[cluster_group == type].mean(axis=0))

    return np.array(new_cens)

def mykmeans(X, no_of_clusters = 3, epochs = 1000):
  centroids = X[random.sample(range(0,X.shape[0]),no_of_clusters)]  ## taking the random centroids
  cluster_group = None
  ## iterating in epochs
  for i in range(epochs):
        cluster_group = group_clusters(X, centroids)
        old_centroids = centroids

        # calculating the new centroid points
        centroids = new_centroids(X,cluster_group)
        if (old_centroids == centroids).all():
            print('Run Completed! at epoch : ', i)
            break

  return cluster_group,centroids ## returning the cluster_group and centroids

y_pred,cluster_centroids = mykmeans(rgb_array,5)  ## running model for k = 5

cluster_centroids

rgb_data = rgb_array.copy()

"""### Task - C"""

from PIL import Image

for i in np.unique(y_pred):
    rgb_data[y_pred == i] = cluster_centroids[i]

def show_image(image_array):
  img_compressed = image_array.reshape(height, width , channels)
  # creating a figure with two subplots
  fig, axes = plt.subplots(1, 2, figsize=(10, 5))

  # displaying the first image on the left subplot
  axes[0].imshow(rgb_image)
  axes[0].set_title('Original Image')

  # displaying the second image on the right subplot
  axes[1].imshow(img_compressed)
  axes[1].set_title('Compressed')

  # hiding the axes
  for ax in axes:
      ax.axis('off')

  # showing the plot
  plt.show()

show_image(rgb_data)  ## displaying the business

## running k-means for different value of k
k_vals = np.arange(1, 10, 2)
for i in k_vals:
    y_pred,cluster_centroids = mykmeans(rgb_array,i,250)
    rgb_data = rgb_array.copy()
    for j in np.unique(y_pred):
        rgb_data[y_pred == j] = cluster_centroids[j]
    print(f'value of k is {i}')
    show_image(rgb_data)

## function to calculate the WCSS
def WCSS(rgb_array, centroids, labels_):
   wcss = 0
   for centroid_idx, centroid in enumerate(centroids):
       cluster_samples = rgb_array[labels_ == centroid_idx]
       cluster_wcss = np.sum((cluster_samples - centroid) ** 2)
       wcss += cluster_wcss
   return wcss

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

k_values = range(2, 10)
wcss_scores = []
# Performing the elbow method for the K-Means
for k in k_values:
    y_pred,cluster_centroids = mykmeans(rgb_array,k)
    wcss = WCSS(rgb_array, cluster_centroids, y_pred)
    wcss_scores.append(wcss)

# Plotting WCSS scores
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(k_values, wcss_scores, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method')

plt.tight_layout()
plt.grid(True)
plt.show()

## from the graph we can see that "k = 6" should be the better choice

## running the k-means for k = 6
y_pred,cluster_centroids = mykmeans(rgb_array,6)
kmeans_centroids = cluster_centroids
kmeans_y_pred  = y_pred

## calculate the percentage of colors present in compressed image
percent=[]
labels = list(kmeans_y_pred)
for i in range(len(kmeans_centroids)):
  j=labels.count(i)
  j=j/(len(kmeans_y_pred))
  percent.append(j)
print(percent)

## plotting the pie chart
plt.pie(percent,colors=np.array(kmeans_centroids/255),labels=np.arange(len(kmeans_centroids)), autopct='%1.1f%%', startangle=140)
plt.title("Dominant Colors in Compressed Image using KMeans")
plt.show()

"""### Task - D(Implementing Using SKlearn)"""

from sklearn.cluster import KMeans  ## importing the KMeans

lst = list(np.arange(1,10, 2))

## running the k-means for sklearn
k_vals = np.arange(1, 10, 2)
md = []
for i in k_vals:
    kmeans = KMeans(n_clusters = i, n_init="auto")
    s = kmeans.fit(rgb_array)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    md.append(kmeans.inertia_)
    rgb_data = rgb_array.copy()
    for j in np.unique(labels):
        rgb_data[labels == j] = centroids[j]
    print(f'value of k is {i}')
    show_image(rgb_data)

## plotting the inertia vs no_of_clusters
plt.plot(lst ,md)
plt.xlabel("No of Clusters")
plt.ylabel("Inertia")
plt.title("Inertia vs No of Clusters")
plt.grid(True)
plt.show()

## from sklearn also we can see that "k = 6" , would be better choice

## plotting the pie chart of dominant colors using sklearn
kmeans=KMeans(n_clusters=6, n_init="auto")
kmeans.fit(rgb_array)
labels=kmeans.labels_
centroid=kmeans.cluster_centers_
labels=list(labels)
percent=[]
for i in range(len(centroid)):
  j=labels.count(i)
  j=j/(len(labels))
  percent.append(j)
plt.pie(percent,colors=np.array(centroid/255),labels=np.arange(len(centroid)), autopct='%1.1f%%', startangle=140)
plt.title("Dominant Colors in the Compressed Image Using Sklearn")
plt.show()

"""### Task - E"""

from PIL import Image
import numpy as np
import random

"""
Code to extract the pixels along with the spatial coordinates
"""
image_path = 'test.png'
image = Image.open(image_path)

image = image.convert('RGB')

width, height = image.size

rgb_values = []
for y in range(height):
    for x in range(width):
        r, g, b = image.getpixel((x, y))
        rgb_values.append((r, g, b, y, x))
rgb_array_spatial = np.array(rgb_values)
print(rgb_array_spatial.shape)



def assign_clusters_sp_coh(X, centroids, spatial_weight=0.5):
    group = []
    # calculate the RGB distances
    rgb_distances = np.linalg.norm(X[:, np.newaxis, :3] - centroids[:, :3], axis=2)
    # calculate spatial distances using the last two columns (coordinates)
    spatial_distances = np.linalg.norm(X[:, np.newaxis, 3:] - centroids[:, 3:], axis=2)
    # combine RGB and spatial distances with the specified weight
    combined_distances = rgb_distances + spatial_weight * spatial_distances
    # assigning clusters based on the minimum combined distance
    group = np.argmin(combined_distances, axis=1)

    return group

def move_centroids_sp_coh(X, cluster_group):
    new_centroids = []

    cluster_type = np.unique(cluster_group)

    for cluster_idx in cluster_type:
        new_centroids.append(X[cluster_group == cluster_idx].mean(axis=0))

    return np.array(new_centroids)

def mykmeans_sp_coh(X, n_clusters, spatial_weight=0.5):
    max_itr = 500
    centroids = X[random.sample(range(0, X.shape[0]), n_clusters)]

    for i in range(max_itr):
        # assigning clusters
        cluster_group = assign_clusters_sp_coh(X, centroids, spatial_weight)
        old_centroids = centroids

        # moving centroids
        centroids = move_centroids_sp_coh(X, cluster_group)

        # checking convergence
        if (old_centroids == centroids).all():
            break

    return cluster_group, centroids

labels, group = mykmeans_sp_coh(rgb_array_spatial, 5)

rgb_data_2 = rgb_array_spatial[:,:3].copy()
for j in np.unique(labels):
    rgb_data_2[labels == j] = group[:,:3][j]
show_image(rgb_data_2)

def show_image_coherence(image_array0, image_array1, image_array2):
  img_compressed0 = image_array0.reshape(height, width , channels)
  img_compressed1 = image_array1.reshape(height, width , channels)
  img_compressed2 = image_array2.reshape(height, width , channels)
  # Create a figure with two subplots
  fig, axes = plt.subplots(1, 3,  figsize=(10, 5))

  # Display the first image on the left subplot
  axes[1].imshow(img_compressed1)
  axes[1].set_title('KMeans Compressed')

  # Display the second image on the right subplot
  axes[2].imshow(img_compressed2)
  axes[2].set_title('KMeans with Coherence Compressed')

  axes[0].imshow(img_compressed0)
  axes[0].set_title('Original Image')
  # Hide the axes
  for ax in axes:
      ax.axis('off')

  # Show the plot
  plt.show()

k_vals = np.arange(2, 10, 2)
md = []
for i in k_vals:
    kmeans = KMeans(n_clusters = i, n_init="auto")
    s = kmeans.fit(rgb_array)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    md.append(kmeans.inertia_)
    rgb_data = rgb_array.copy()

    labels1, group = mykmeans_sp_coh(rgb_array_spatial, i)
    rgb_data_2 = rgb_array_spatial[:,:3].copy()
    for j in np.unique(labels1):
        rgb_data_2[labels1 == j] = group[:,:3][j]
    # show_image(rgb_data_2)

    for j in np.unique(labels):
        rgb_data[labels == j] = centroids[j]
    print(f'value of k is {i}')
    show_image_coherence(rgb_array, rgb_data, rgb_data_2)

labels1, centroid = mykmeans_sp_coh(rgb_array_spatial, 6)
labels1=list(labels1)
percent=[]
for i in range(len(centroid)):
  j=labels1.count(i)
  j=j/(len(labels1))
  percent.append(j)
plt.pie(percent,colors=np.array(centroid/255),labels=np.arange(len(centroid)), autopct='%1.1f%%', startangle=140)
plt.title("Dominant Colors in the Compressed Image Using Spatial Coherence")
plt.show()