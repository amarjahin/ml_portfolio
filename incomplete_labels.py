import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linear_models import LinearRegression 

# the data has 4 columns. The first is the true label, the second and third are the features, 
# the last is the incomplete label. We need to predict the true label using 
# the features and the incomplete label.

data_train = pd.read_csv('data/ds3_train.csv')
data_valid = pd.read_csv('data/ds3_valid.csv')
data_test = pd.read_csv('data/ds3_test.csv')

x_train = data_train.iloc[:, 1:3].values
y_train = data_train.iloc[:, -1].values
t_train = data_train.iloc[:, 0].values

x_valid = data_valid.iloc[:, 1:3].values
y_valid = data_valid.iloc[:, -1].values
t_valid = data_valid.iloc[:, 0].values

x_test = data_test.iloc[:, 1:3].values
t_test = data_test.iloc[:, 0].values


lr = LinearRegression(num_features=2)
# train the model using incomplete labels 
lr.fit(x_train, y_train)
y_pred = lr.predict(x_valid)
print("The accuracy on validating set incomplete labels is: ", np.mean(y_pred == y_valid))

# rescale the desision threshold to get the true labels
m, n = x_valid.shape
x = np.concatenate((np.ones((m, 1)), x_valid), axis=1)
alpha = np.sum(y_valid * lr.h(x)) / np.sum(y_valid)
print("The alpha is: ", alpha)
t_pred = lr.predict(x_valid, threshold=alpha*0.5)
print("The accuracy on validating set true labels is: ", np.mean(t_pred == t_valid))
t_pred = lr.predict(x_test, threshold=alpha*0.5)
print("The accuracy on test set true labels is: ", np.mean(t_pred == t_test))


x1_min, x1_max = x_test[:, 0].min(), x_test[:, 0].max()
x2_min, x2_max = x_test[:, 1].min(), x_test[:, 1].max()
x1_line = np.linspace(x1_min*0.8, x1_max*1.2, 100)
x2_line = (-(lr.theta[0] + lr.theta[1] * x1_line + np.log(2/alpha - 1)) / (lr.theta[2] + 1j*1e-16)).real 
plt.scatter(x_test[:, 0], x_test[:, 1], c=t_test)
plt.plot(x1_line, x2_line, 'r-', linewidth=2)
plt.legend(['Test set true labels', 'Logistic Regression'])
plt.xlabel(r'$x_1$')    
plt.ylabel(r'$x_2$')
plt.title('Decision boundaries logistic regression')
plt.grid(True, alpha=0.3)
plt.show()
