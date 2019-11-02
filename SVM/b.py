# Author: Wesley Watkins
# wjw16

# Import and Initializations
import pandas as pd
from svm import SupportVectorMachine

# import data generator
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
import data as d

# Generate sample data
print("\nGenerating training data...")
X, Y = d.gen_data()
print("Training data generated!")

# Create and train support vector machine
svm = SupportVectorMachine()
print('\nTraining support vector machine...')
svm.train(X, Y, 5)
print('Training complete!')

# plot the training points and decision boundary
print('\nPlotting Decision Boundary...')
print("Plotted!")
svm.visualize()

# Get Info From SVM
print('\nCalculating margin length...')
print('Margin Length:', round(svm.getMargin(), 4))

print("\nFinding support vectors...")
sv = svm.getSupportVectors()
print("Support Vectors ( Count =", len(sv), "):")
for row in sv:
    print("(" + str(round(row[0], 7)) + ", " + str(round(row[1], 7)) + ")")

print("\nCalculating leave one out cross validation error...")
print("Leave one out error =", round(svm.getLOOE(1), 4))

# Begin Test Phase
print("\nGenerating test data...")
X2, Y2 = d.gen_data(10430)
print("Test data generated!")

# Testing different values of C
print("\nMisclassification error with different values of C...")
values = svm.testVariety(X2, Y2)
for i in range(values.shape[0]):
    values[i, 1] = round(values[i, 1]/Y2.size * 100, 4)
df = pd.DataFrame(values, columns=list(["C", "Error %"]))
print(df)

# Testing different values of C
print("\nMargin with different values of C...")
values = svm.getMarginVariety()
df = pd.DataFrame(values, columns=list(["C", "Margin"]))
print(df)

print('\nPlotting Decision Boundary for different values of C...')
print("Plotted!")
svm.visualizeVariety()
