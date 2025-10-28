import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay

# Load the data
data_train = pd.read_csv('data/ds7_train.csv')
data_valid = pd.read_csv('data/ds7_test.csv')

x_train = data_train.iloc[:, 1:3].values
y_train = data_train.iloc[:, 0].values
x_valid = data_valid.iloc[:, 1:3].values
y_valid = data_valid.iloc[:, 0].values

clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)


y_pred = clf.predict(x_valid)

print("The accuracy on validating set is: ", np.mean(y_pred == y_valid))

# plt.scatter(x_valid[:, 0], x_valid[:, 1], c=y_valid)
fig, ax = plt.subplots()
ax.scatter(x_valid[:, 0], x_valid[:, 1], c=y_valid)
common_params = {"estimator": clf, "X": x_train, "ax": plt.gca()}
DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
)
DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Validating set, decision boundary and margins: SVM with RBF Kernel')
plt.grid(True, alpha=0.3)
plt.show()
