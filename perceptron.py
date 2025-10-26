import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linear_models import Perceptron

data_train = pd.read_csv('data/ds2_train.csv')
data_valid = pd.read_csv('data/ds2_valid.csv')

x_train = data_train.iloc[:, :-1].values
y_train = data_train.iloc[:, -1].values
x_valid = data_valid.iloc[:, :-1].values
y_valid = data_valid.iloc[:, -1].values

plt.scatter(x_train[:,0], x_train[:,1], c=y_train)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Training set')
plt.show()


pp = Perceptron(num_features=2)
pp.fit(x_train, y_train, learning_rate=1, eps=1e-9, max_iter=10000)

y_pred = pp.predict(x_valid)



print("The accuracy on training set is using perceptron: ", np.mean(pp.predict(x_train) == y_train))
print("The accuracy on validating set is using perceptron: ", np.mean(pp.predict(x_valid) == y_valid))


plt.scatter(x_valid[:,0], x_valid[:,1], c=y_valid)
plt.xlabel('x1')
plt.ylabel('x2')

x1_min, x1_max = x_valid[:, 0].min(), x_valid[:, 0].max()
x2_min, x2_max = x_valid[:, 1].min(), x_valid[:, 1].max()
x1_line = np.linspace(x1_min*0.8, x1_max*1.2, 100)

# the imaginary part is to avoid division by zero
x2_line = (-(pp.theta[0] + pp.theta[1] * x1_line) / (pp.theta[2] + 1j*1e-16)).real 
plt.plot(x1_line, x2_line, 'r-', linewidth=2)

plt.legend(['Validating Set', 'Perceptron'])
plt.xlabel(r'$x_1$')    
plt.ylabel(r'$x_2$')
plt.title('Decision Boundaries, Perceptron')
plt.grid(True, alpha=0.3)
plt.show()