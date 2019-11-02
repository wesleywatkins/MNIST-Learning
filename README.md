# CIS4930-PA1

## Libraries Used
- cvxpy (for convex optimization problem)
- matplotlib (for plotting results)
- pandas (for printing arrays all pretty-like)
- mlxtend (for reading in MNIST dataset)
- numpy (for obvious reasons)

## How To Run Files
- Install the dependencies listed above
- Any files labeled "a.py", "b.py", and "c.py" can each be ran directly
- The seed for training data created in "./data.py" is 19719 (last 5 of my lib card #)
- The seed for test data created in "./data.py" is 10430 (honestly, just random)

## Files
- **./data.py** : generate data as specified in project description
- **./SVM/svm.py** : creates an Support Vector Machine model class
- **./SVM/b.py** : trains SVM with data generated from "./data.py", plots/displays decision boundary, computes and prints margin length, support vectors, cross-validation error, and various margin lengths and misclassification errors for different values of C, and finally plots decision boundaries for different values of C
- **./SVM/c.py** : reads in MNIST training and test data sets, trains SVM on training data, then computes generalization error on test data
- **./Regression/linear.py** : creates a linear regression model class
- **./Regression/logistic.py** : creates a logistic regression model class
- **./Regression/a.py** : trains linear model with data generated from "./data.py", calculates generalization error on test data generated in the same manner, computes the leave one out validation error, and plots the decision boundary
- **./Regression/b.py** : trains logistic model with data generated from "./data.py", calculates generalization error on test data generated in the same manner, computes the leave one out validation error, and plots the decision boundary
- **./Regression/c.py** : reads in MNIST training and test data sets, trains SVM, linear, and logistic regression models on training data, then computes generalization error on test data for each model
