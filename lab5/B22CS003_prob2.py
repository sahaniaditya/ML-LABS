"""# PART - 2 (SUPPORT VECTOR MACHINE)

### Task - 1(a)
"""

## importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## loading the datasets
from sklearn import datasets
iris = datasets.load_iris(as_frame=True)

## extracting the data out from iris
X = iris.data
y = iris.target

iris.target_names  ## printing the target names

print("Shape of y : ",y.shape) ## printing the shape of y

print("Shape of X : ",X.shape) ## printing the shape of X

## converting X and y to array
y = np.array(y)
X = np.array(X)
y = y.reshape(-1,1)  ## reshaping the y

data = np.concatenate((X,y), axis=1)  ## concatenating X and y

df = pd.DataFrame(data, columns=["sepal length (cm)", "sepal width (cm)","petal length (cm)", "petal width (cm)", "target"])  ## creating a dataframe

df.info() ## printing the info about datasets

new_df1 = df[df["target"] == 0.00]
new_df2 = df[df["target"] == 1.00]
new_df = pd.concat((new_df1, new_df2))

new_df.head() ## new_df after merging the two above

final_df = new_df[["petal length (cm)", "petal width (cm)", "target"]]  ## final dataset after selecting only two features

final_df.head()

X = final_df.drop(columns=["target"]).values
y = final_df["target"].values

from sklearn.preprocessing import StandardScaler
## importing standard_scaler to scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

## splitting the data in the ratio of 80 : 20
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""### Task - 1(b)"""

from sklearn.svm import LinearSVC  ## importing the linear SVM

## training the linear SVM
svc = LinearSVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)  ## making predictions

from sklearn.metrics import accuracy_score

print("Accuracy : ",accuracy_score(y_pred, y_test))

## defining the parameter to plot the decision boundary
w = svc.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2, 2)
yy = a * xx - (svc.intercept_[0])/w[1]

from matplotlib.colors import ListedColormap

## Plotting the decision boundary for training and test data

## Training Data
plt.plot(xx, yy)
plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.title("Decision Boundary on Training Data")
plt.grid(True)
plt.show()

## Test Data
plt.plot(xx, yy)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.title("Decision Boundary on Test Data")
plt.grid(True)
plt.show()

"""### Task - 2(a)"""

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=500, noise=0.05, random_state=42)  ## generating dataset using make_moons()

## adding some percentage of misclassifications points
num_noise_points = int(0.05 * len(X))
random_indices = np.random.choice(len(X), num_noise_points, replace=False)
y[random_indices] = 1 - y[random_indices] ## flipping the label points

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## plotting the dataset
plt.scatter(X[:,0], X[:,1], c=y)
plt.grid(True)
plt.title("Make Moons Dataset")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

"""### Task - 2(b)"""

from sklearn.svm import SVC

## initializing teh different kernel models
svm_linear = SVC(kernel='linear', random_state=42)
svm_poly = SVC(kernel='poly', degree=5, random_state=42)
svm_rbf = SVC(kernel='rbf', gamma=0.9, random_state=42)

## fitting the different SVM models
svm_linear.fit(X, y)
svm_poly.fit(X, y)
svm_rbf.fit(X, y)

## plotting the decision boundary for different SVMs
def plot_decision_boundary(svm_model, title):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                         np.linspace(ylim[0], ylim[1], 100))
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.title(title)
    plt.show()


plot_decision_boundary(svm_linear, 'Linear Kernel Plot')
plot_decision_boundary(svm_poly, 'Polynomial Kernel Plot')
plot_decision_boundary(svm_rbf, 'RBF Kernel Plot')

"""### Task - 2(c)"""

## Performing the GridSearchCV and RandomizedSearchCV for the "RBF" kernels

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import reciprocal, uniform
## defining the parameters
c_lst = [0.1, 0.5, 0.9, 1, 5, 10, 20, 30, 50, 100]
g_lst = [0.001, 0.05, 0.07, 0.1, 0.5, 0.7, 0.9, 1]
param_grid = {'C': c_lst, 'gamma': g_lst}
## initializing the SVM using "rbf"
svm_rbf_kernel = SVC(kernel='rbf')

## performing the GridSearchCV
grid_search = GridSearchCV(svm_rbf_kernel, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

## printing the best parameters
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

## evaluating the model accuracy
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test set accuracy:", test_score)

## defining the parameters using reciprocal distribution
param_dist = {
    'C': reciprocal(0.1, 100),
    'gamma': reciprocal(0.001, 1)
}

# initializing the SVM with RBF kernel
svm = SVC(kernel='rbf')

# performing the random search
random_search = RandomizedSearchCV(svm, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

# printing the best parameters
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)

# evaluating the model and printing accuracy
best_model = random_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test set accuracy:", test_score)

"""### Task - 2(d)"""

## best parameters from the GridSearchCV
best_params = {'C': 0.5, 'gamma': 0.9}

# training the svm for best parameters
best_svm = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
best_svm.fit(X_train, y_train)

# Plotting decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k')

plt.figure(figsize=(8, 4))
plot_decision_boundary(best_svm, X_train, y_train)
plt.xlabel('Petal length(cm)')
plt.ylabel('Petal width(cm)')
plt.title('Plot Using Best Parameters')
plt.show()

## best parameters from the RandomSearchCV
best_params = {'C': 1.3292918943162166, 'gamma': 0.711447600934342}

# training the svm for best parameters
best_svm = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
best_svm.fit(X_train, y_train)

# Plotting decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k')

plt.figure(figsize=(8, 4))
plot_decision_boundary(best_svm, X_train, y_train)
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Plot Using Best Parameters')
plt.show()

