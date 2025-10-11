import numpy as np

class LinearRegression():
    """
    Logistical regression model. 
    Parameters:
        num_features (int): number of features in the data
        eps (float): convergence threshold for Newton's method
    """

    def __init__(self, num_features, eps=1e-9):
        self.num_features = num_features
        self.eps = eps
        self.theta = np.zeros(num_features+1)

    def h(self, x):
        """
        Logistical regression hypothesis function, or the probability of the positive class. 
        h(x) = 1 / (1 + exp(-x @ theta)) where x is the input data and theta is the model parameters.
        Parameters:
            x (numpy.ndarray): input data
        Returns:
            numpy.ndarray: hypothesis function 0<=h(x)<=1
        """
        z = x @ self.theta
        return 1 / (1 + np.exp(-z))
    def log_likelihood(self, x, y):
        """
        log the likelihood of the parameters theta, log(P(y|x;theta)). 
        Parameters:
            x (numpy.ndarray): input data
            y (numpy.ndarray): target data
        Returns:
            float: log likelihood
        """
        return -np.mean(y * np.log(self.h(x)) + (1 - y) * np.log(1 - self.h(x)))
    def gradient(self, x, y):
        """
        Gradient of the log likelihood function.
        Parameters:
            x (numpy.ndarray): input data
            y (numpy.ndarray): target data
        Returns:
            numpy.ndarray: gradient
        """
        return -np.mean(x * (y - self.h(x))[:,None], axis=0)
    def hessian(self, x):
        """
        Hessian of the log likelihood function.
        Parameters:
            x (numpy.ndarray): input data
        Returns:
            numpy.ndarray: Hessian
        """
        return np.mean(x[:,:,None] * x[:,None,:] * (self.h(x) * (1 - self.h(x)))[:,None,None], axis=0)
    def next_theta(self, x, y):
        """
        Update theta using Newton's method.
        Parameters:
            x (numpy.ndarray): input data
            y (numpy.ndarray): target data
        Returns:
            numpy.ndarray: next theta
        """
        return self.theta - np.linalg.inv(self.hessian(x)) @ self.gradient(x, y)

    def fit(self, x, y):
        """
        Keep updating theta until the change is less than the convergence threshold.
        Parameters:
            x (numpy.ndarray): input data
            y (numpy.ndarray): target data
        """
        m, n = x.shape
        x = np.concatenate((np.ones((m, 1)), x), axis=1)

        old_theta = self.theta
        self.theta = self.next_theta(x, y)
        while np.linalg.norm(self.theta - old_theta, 1) >= self.eps:
            old_theta = self.theta
            self.theta = self.next_theta(x, y)

    def predict(self, x, threshold=0.5):
        """
        Predict the class of the input data, using the current theta. 
        h(x) >= 0.5 is the positive class, otherwise the negative class.
        You should train the model first before predicting.
        Parameters:
            x (numpy.ndarray): input data
        Returns:
            numpy.ndarray: predicted class
        """
        m, n = x.shape
        x = np.concatenate((np.ones((m, 1)), x), axis=1)
        return self.h(x) >= threshold


class GDA():
    def __init__(self, num_features):
        self.num_features = num_features
        self.theta = np.zeros(num_features+1)

    def fit(self, x, y):
        self.phi = np.mean(y)
        self.mu = np.array([np.mean(x[y == 0], axis=0), np.mean(x[y == 1], axis=0)])
        x_shifted = np.concatenate((x[y==0] - self.mu[0][None,:], x[y==1] - self.mu[1][None,:]), axis=0)
        self.sigma = np.cov(x_shifted.T)
        self.sigma_inv = np.linalg.inv(self.sigma)
        self.theta[1:] = self.sigma_inv @ (self.mu[1] - self.mu[0]).T
        self.theta[0] =  np.log(self.phi / (1 - self.phi)) + (self.mu[0] @ self.sigma_inv @ self.mu[0].T - self.mu[1] @ self.sigma_inv @ self.mu[1].T)/2
    
    def h(self, x):
        z = x @ self.theta
        return 1 / (1 + np.exp(-z))
    
    def predict(self, x):
        m, n = x.shape
        x = np.concatenate((np.ones((m, 1)), x), axis=1)
        return self.h(x) >= 0.5