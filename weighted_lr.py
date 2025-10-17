import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linear_models import LinearRegression

# Load the data
data_train = pd.read_csv('data/ds5_train.csv')
data_valid = pd.read_csv('data/ds5_valid.csv')

x_train = data_train.iloc[:, :-1].values
y_train = data_train.iloc[:, -1].values
x_valid = data_valid.iloc[:, :-1].values
y_valid = data_valid.iloc[:, -1].values

tau_list = [0.1, 0.3, 0.5]
y_pred = np.zeros((len(x_valid), len(tau_list))) 
lr = LinearRegression(num_features=2)
for i, x in enumerate(x_valid):
    for j, tau in enumerate(tau_list):
        lr.fit(x_train, y_train, weights="Gaussian", x_to_predict=x[0], sigma=tau)
        y_pred[i, j] = lr.predict(x[0])

plt.scatter(x_valid[:,0], y_valid, label='Validation set')
for j in range(len(tau_list)):
    plt.scatter(x_valid[:,0], y_pred[:,j], label=f'tau={tau_list[j]}')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with Gaussian Weights and different sigma values')
plt.show()