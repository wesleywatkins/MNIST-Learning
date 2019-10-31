# Author: Wesley Watkins
# wjw16

# Import and Initializations
import data as d
import random
import numpy as np
import pandas as pd
from svm import SupportVectorMachine


# Generate sample data
print("\nGenerating training data...")
X, Y = d.gen_data()
print("Training data generated!")

# Create and train support vector machine
svm = SupportVectorMachine()
print('\nTraining support vector machine...')
svm.train(X, Y, 200)
print('Training complete!')
print("Optimal w: [ ", end="")
for i in range(svm.w.shape[0]):
    print(round(svm.w[i].item(), 7), end=" ")
print("]")
print("Optimal b:", round(svm.b.item(), 7))
print("Chosen C:", round(svm.C, 7))

# plot the training points and decision boundary
print('\nPlotting Decision Boundary...')
print("Plotted!")
svm.visualize()

# Get Info From SVM
print('\nInfo About SVM:')
print('Margin Length:', round(svm.getMargin(), 4))
print("Leave One Out Error <=", 100 * round(svm.getLOOE(), 2), "%")
sv = svm.getSupportVectors()
np.set_printoptions(precision=3)
print("Support Vectors ( Count =", len(sv), "):")
for row in sv:
    print("(" + str(round(row[0], 7)) + ", " + str(round(row[1], 7)) + ")")

# Begin Test Phase
print("\nGenerating test data...")
X2, Y2 = d.gen_data(10430)
print("Test data generated!")

# Testing different values of C
print("\nMisclassification error with different values of C...")
values = svm.test(X2, Y2, diff_Cs=True)
for i in range(values.shape[0]):
    values[i, 1] = round(values[i, 1]/Y2.size * 100, 4)
df = pd.DataFrame(values, columns=list(["C", "Error %"]))
print(df)

# Testing different values of C
print("\nMargin with different values of C...")
values = svm.getMargin(diff_Cs=True)
df = pd.DataFrame(values, columns=list(["C", "Margin"]))
print(df)

print('\nPlotting Decision Boundary for different values of C...')
print("Plotted!")
svm.visualize(diff_Cs=True)
