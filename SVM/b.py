# Author: Wesley Watkins
# wjw16

# Import and Initializations
import numpy as np
import matplotlib.pyplot as plt
import data as d
import random
from svm import SupportVectorMachine


# Generate sample data
print("Generating training data...")
X, Y = d.gen_data()
print("Training data generated!")

# Create support vector machine
svm = SupportVectorMachine()
print('\nTraining support vector machine...')
svm.train(X, Y, 100)
print('Training complete!')
print("Optimal w:", svm.w)
print("Optimal b:", svm.b)
print("Chosen C:", svm.C)

# Begin Test Phase
print("\nGenerating test data...")
X2, Y2 = d.gen_data(random.randint(0, 999999999))
print("Test data generated!")

# Predict and Figure Out Accuracy
misclassified = 0
for i in range(0, Y2.size):
    prediction = svm.predict(X2[i])
    actual = Y2[i]
    if prediction != actual:
        misclassified += 1
print("Misclassified:", misclassified, "/", Y2.size)
print("Accuracy (on test data):", (1 - (misclassified/Y2.size)) * 100, '%')

# Get Info From SVM
print('\nInfo About SVM:')
print('Margin Length:', round(svm.getMargin(), 4))
print("Leave One Out Error <=", 100 * round(svm.getLOOE(), 2), "%")
print("Support Vectors ( Count =", len(svm.getSupportVectors()), "): \n", svm.getSupportVectors())

print('\nPlotting Decision Boundary...\n')
svm.visualize()
