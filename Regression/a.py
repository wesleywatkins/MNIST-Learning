import data as d
import random
from linear import LinearRegression

print("Generating training data...")
X, Y = d.gen_data()
print("Training data generated!")

# Create support vector machine
linear = LinearRegression()
print('\nComputing best values for w0 and w1...')
linear.train(X, Y)
print('Computation complete!')

print("\nGenerating (truly random) test data...")
X2, Y2 = d.gen_data(10430)
print("Test data generated!")

# Predict and Figure Out Accuracy
misclassified = linear.test(X2, Y2)
print("Misclassified:", misclassified, "/", Y2.size)
print("Accuracy (on test data):", (1 - (misclassified/Y2.size)) * 100, '%')
print("Leave One Out Error =", round(linear.getLOOE(), 4))

# Plot the graph
print("\nPlotting points and decision boundary")
linear.visualize()
