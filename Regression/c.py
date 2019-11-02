# Wesley Watkins
# wjw16

# imports
from linear import LinearRegression
from logistic import LogisticRegression
from mlxtend.data import loadlocal_mnist
import os
import numpy as np

# Import SVM (avoid having two create two copies of SVM)
import sys
sys.path.append("../SVM/")  # Adds higher directory to python modules path.
from svm import SupportVectorMachine

# read in MNIST dataset
print("Reading in MNIST dataset...")
dir_path = os.path.dirname(os.path.realpath(__file__))
X_train, Y_train = loadlocal_mnist(
    images_path=os.path.join(dir_path, '../Samples/train-images.idx3-ubyte'),
    labels_path=os.path.join(dir_path, '../Samples/train-labels.idx1-ubyte')
)
X_test, Y_test = loadlocal_mnist(
    images_path=os.path.join(dir_path, '../Samples/t10k-images.idx3-ubyte'),
    labels_path=os.path.join(dir_path, '../Samples/t10k-labels.idx1-ubyte')
)
print("Reading complete!")

# keep only 0s and 1s
print("\nRemoving non 0-1 labels from training dataset...")
indexes = list()
for i in range(Y_train.size):
    if Y_train[i] == 0 or Y_train[i] == 1:
        indexes.append(i)
X = np.zeros(shape=(len(indexes), X_train.shape[1]))
Y = np.zeros((len(indexes), 1))
for (i, index) in enumerate(indexes):
    X[i] = X_train[index]
    if Y_train[index] == 0:
        Y[i, 0] = -1
    else:
        Y[i, 0] = Y_train[index]
X_train = None
Y_train = None
print("Non 0-1 labels removed from training dataset!")

# keep only 0s and 1s
print("\nRemoving non 0-1 labels from testing dataset...")
indexes = list()
for i in range(Y_test.size):
    if Y_test[i] == 0 or Y_test[i] == 1:
        indexes.append(i)
X2 = np.zeros(shape=(len(indexes), X_test.shape[1]))
Y2 = np.zeros((len(indexes), 1))
for (i, index) in enumerate(indexes):
    X2[i] = X_test[index]
    if Y_test[index] == 0:
        Y2[i, 0] = -1
    else:
        Y2[i, 0] = Y_test[index]
# clear up memory
X_test = None
Y_test = None
print("Non 0-1 labels removed from testing dataset!")

print("\nTraining SVM on MNIST dataset...")
svm = SupportVectorMachine()
svm.train(X, Y, 1)
print("SVM trained!")

print("\nTraining Linear Regression on MNIST dataset...")
linear = LinearRegression()
linear.train(X, Y)
print("Linear regression trained!")

print("\nTraining Logistic Regression on MNIST dataset...")
logistic = LogisticRegression()
logistic.train(X, Y)
print("Logistic regression trained!")

# Test SVM
print("\nRunning SVM on test data...")
misclassified = svm.test(X2, Y2)
print("Generalization Error:", round(misclassified/Y2.size, 3))
print("Misclassified:", misclassified, "/", Y2.size)
print("Accuracy (on test data):", round((1 - (misclassified/Y2.size)) * 100, 3), '%')

# Test Linear Regression
print("\nRunning Linear Regression on test data...")
misclassified = linear.test(X2, Y2)
print("Generalization Error:", round(misclassified/Y2.size, 3))
print("Misclassified:", misclassified, "/", Y2.size)
print("Accuracy (on test data):", round((1 - (misclassified/Y2.size)) * 100, 3), '%')

# Test Logistic Regression
print("\nRunning Logistic Regression on test data...")
misclassified = logistic.test(X2, Y2)
print("Generalization Error:", round(misclassified/Y2.size, 3))
print("Misclassified:", misclassified, "/", Y2.size)
print("Accuracy (on test data):", round((1 - (misclassified/Y2.size)) * 100, 3), '%')