import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linear_models import LinearRegression, GDA

# Load the data
data_train = pd.read_csv('data/ds1_train.csv')
data_valid = pd.read_csv('data/ds1_valid.csv')

x_train = data_train.iloc[:, :-1].values
y_train = data_train.iloc[:, -1].values
x_valid = data_valid.iloc[:, :-1].values
y_valid = data_valid.iloc[:, -1].values

# Train the models
lr = LinearRegression(num_features=2)
lr.fit(x_train, y_train)
gda = GDA(num_features=2)
gda.fit(x_train, y_train)

# Evaluate the models
print("The accuracy on training set is: ", np.mean(lr.predict(x_train) == y_train))
print("The accuracy on validating set is: ", np.mean(lr.predict(x_valid) == y_valid))
print("The accuracy on training set is: ", np.mean(gda.predict(x_train) == y_train))
print("The accuracy on validating set is: ", np.mean(gda.predict(x_valid) == y_valid))

# Plot the decision boundaries
# plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.scatter(x_valid[:, 0], x_valid[:, 1], c=y_valid)

x1_min, x1_max = x_valid[:, 0].min(), x_valid[:, 0].max()
x2_min, x2_max = x_valid[:, 1].min(), x_valid[:, 1].max()
x1_line = np.linspace(x1_min*0.8, x1_max*1.2, 100)

# the imaginary part is to avoid division by zero
x2_line = (-(lr.theta[0] + lr.theta[1] * x1_line) / (lr.theta[2] + 1j*1e-16)).real 
plt.plot(x1_line, x2_line, 'r-', linewidth=2)

x2_line = (-(gda.theta[0] + gda.theta[1] * x1_line) / (gda.theta[2] + 1j*1e-16)).real
plt.plot(x1_line, x2_line, 'b-', linewidth=2)

plt.legend(['Validating Set', 'Logistic Regression', 'GDA'])
plt.xlabel(r'$x_1$')    
plt.ylabel(r'$x_2$')
plt.xlim(x1_min*0.6, x1_max*1.1)
plt.ylim(x2_min - 100, x2_max*1.1)
plt.title('Decision Boundaries, logistic regression and GDA')
plt.grid(True, alpha=0.3)
plt.show()