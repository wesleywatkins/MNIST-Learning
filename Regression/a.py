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
X2, Y2 = d.gen_data(random.randint(0, 999999999))
print("Test data generated!")

# Predict and Figure Out Accuracy
misclassified = 0
for i in range(0, Y2.size):
    prediction = linear.predict(X2[i])
    actual = Y2[i]
    if prediction != actual:
        misclassified += 1
print("Misclassified:", misclassified, "/", Y2.size)
print("Accuracy (on test data):", (1 - (misclassified/Y2.size)) * 100, '%')

# Plot the graph
print("\nPlotting points and decision boundary")
linear.visualize()
