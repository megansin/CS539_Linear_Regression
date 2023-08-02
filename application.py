import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.asmatrix(y).T
X = np.asmatrix(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################
## INSERT YOUR CODE HERE

alphas = [0.001, 0.005, 0.01, 0.05, 0.1]
epochs = [100, 200, 300, 400, 500]


for a in alphas:
    for e in epochs:
        print("alpha:", a, "\nepochs:", e)

        w = train(Xtrain, Ytrain, a, e)
        yhat = Xtrain.dot(w)
        yhat_test = Xtest.dot(w)

        train_loss = compute_L(yhat, Ytrain)
        test_loss = compute_L(yhat_test, Ytest)

        print("Train loss:", train_loss, "\nTest loss:", test_loss)
        print('')


#########################################

