import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linear_models import PerceptronKernel

data_train = pd.read_csv('data/ds7_train.csv')
data_test = pd.read_csv('data/ds7_test.csv')

x_train = data_train.iloc[:, 1:].values
y_train = data_train.iloc[:, 0].values
x_test = data_test.iloc[:, 1:].values
y_test = data_test.iloc[:, 0].values

plt.scatter(x_train[:,0], x_train[:,1], c=y_train)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Training set')
plt.show()


pp = PerceptronKernel(num_features=2, sigma=2)
pp.fit(x_train, y_train, learning_rate=1)


print("The accuracy on training set is using perceptron: ", np.mean(pp.predict(x_train) == y_train))
print("The accuracy on test set is using perceptron: ", np.mean(pp.predict(x_test) == y_test))


x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)
Xtest = np.c_[X1.ravel(), X2.ravel()]
Z = pp.predict(Xtest)
Z = Z.reshape(X1.shape)
plt.scatter(X1, X2, c=Z, alpha=0.1)
plt.scatter(x_test[:,0], x_test[:,1], c=y_test)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Boundary using Perceptron with Gaussian Kernel')
plt.show()