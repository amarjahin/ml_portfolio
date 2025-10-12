import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linear_models import PoissonRegression

data_train = pd.read_csv('data/ds4_train.csv')
data_valid = pd.read_csv('data/ds4_valid.csv')


x_train = data_train.iloc[:, :-1].values
y_train = data_train.iloc[:, -1].values
x_valid = data_valid.iloc[:, :-1].values
y_valid = data_valid.iloc[:, -1].values

# Train the models
pr = PoissonRegression(num_features=4, alpha=1e-7)
pr.fit(x_train, y_train)

y_pred = pr.predict(x_valid)

plt.scatter(y_pred, y_valid)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Poisson Regression')
plt.show()


