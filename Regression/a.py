# import linear regression model
from linear import LinearRegression

# import data generator
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
import data as d

print("\nGenerating training data...")
X, Y = d.gen_data()
print("Training data generated!")

# Create support vector machine
linear = LinearRegression()
print('\nComputing best values for w0 and w1...')
linear.train(X, Y)
print('Computation complete!')

print("\nGenerating test data...")
X2, Y2 = d.gen_data(10430)
print("Test data generated!")

# Predict and Figure Out Accuracy
misclassified = linear.test(X2, Y2)
print("Misclassified:", misclassified, "/", Y2.size)
print("Accuracy (on test data):", (1 - (misclassified/Y2.size)) * 100, '%')

# Compute leave one out error
print("\nComputing leave one out cross validation error...")
print("Leave one out error =", round(linear.getLOOE(), 4))

# Plot the graph
print("\nPlotting points and decision boundary...")
linear.visualize()
