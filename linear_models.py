import numpy as np

class LogisticRegression():
    """
    Logistical regression model. 
    Parameters:
        num_features (int): number of features in the data
        eps (float): convergence threshold for Newton's method
    """

    def __init__(self, num_features):
        self.num_features = num_features
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
    def next_theta_newton(self, x, y):
        """
        Update theta using Newton's method.
        Parameters:
            x (numpy.ndarray): input data
            y (numpy.ndarray): target data
        Returns:
            numpy.ndarray: next theta
        """
        return self.theta - np.linalg.inv(self.hessian(x)) @ self.gradient(x, y)


    def next_theta_gradient(self, x, y, learning_rate):
        return self.theta - learning_rate * self.gradient(x, y)  # plus because we are maximizing the likelihood

    def fit(self, x, y,  eps=1e-9, max_iter=5000, method="Newton", learning_rate=None):
        """
        Keep updating theta until the change is less than the convergence threshold.
        Parameters:
            x (numpy.ndarray): input data
            y (numpy.ndarray): target data
            eps (float): convergence threshold
            max_iter (int): maximum number of iterations
            method (str): method to use, "Newton" or "Gradient Descent"
            learning_rate (float): learning rate for gradient descent
        """
        m, n = x.shape
        x = np.concatenate((np.ones((m, 1)), x), axis=1)

        self.target_is_0_1 = True
        if np.min(y) == -1: # make sure the target data is 0 or 1
            y = (y + np.ones_like(y))/2
            self.target_is_0_1 = False
    
        old_theta = self.theta
        if method == "Newton":
            self.theta = self.next_theta_newton(x, y)
        elif method == "Gradient Descent":
            self.theta = self.next_theta_gradient(x, y, learning_rate)
        i = 0
        while np.linalg.norm(self.theta - old_theta, 1) >= eps and i < max_iter:
            old_theta = self.theta
            if method == "Newton":
                self.theta = self.next_theta_newton(x, y)
            elif method == "Gradient Descent":
                self.theta = self.next_theta_gradient(x, y, learning_rate)
            i += 1

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
        if self.target_is_0_1:
            return self.h(x) >= 0.5
        else:
            return (self.h(x) >= 0.5) * 2 - 1   


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


class PoissonRegression():
    """
    Poisson regression model.
    Parameters:
        num_features (int): number of features in the data
        alpha (float): learning rate for gradient ascent
        eps (float): convergence threshold for gradient ascent (max likelihood)
        max_iter (int): maximum number of iterations
    """
    def __init__(self, num_features,alpha,eps=1e-9, max_iter=5000):
        self.num_features = num_features
        self.eps = eps
        self.theta = np.zeros(num_features+1)
        self.alpha = alpha
        self.max_iter = max_iter
    
    def h(self, x):
        z = x @ self.theta
        # z = np.clip(z, -40, 40)
        return np.exp(z)

    def gradient(self, x, y):
        return np.mean(x * (y - self.h(x))[:,None], axis=0)
    
    # def hessian(self, x):
    #     return -np.mean(x[:,:,None] * x[:,None,:] * self.h(x)[:,None,None], axis=0)
    
    def next_theta(self, x, y):
        return self.theta + self.alpha * self.gradient(x, y)  # plus because we are maximizing the likelihood
    
    def fit(self, x, y):
        m, n = x.shape
        x = np.concatenate((np.ones((m, 1)), x), axis=1)
        old_theta = self.theta
        print(self.theta)
        self.theta = self.next_theta(x, y)
        print(self.theta)
        i = 0
        while np.linalg.norm(self.theta - old_theta, 1) >= self.eps and i < self.max_iter:
            old_theta = self.theta
            self.theta = self.next_theta(x, y)
            print(self.theta)
            i += 1
    def predict(self, x):
        m, n = x.shape
        x = np.concatenate((np.ones((m, 1)), x), axis=1)
        return self.h(x) 


class LinearRegression():
    """
    Linear regression model where training is done using the normal equation. 
    The fitting can be done with or without weights.
    Parameters:
        num_features (int): number of features in the data
    """
    def __init__(self, num_features):
        self.num_features = num_features
        self.theta = np.zeros(num_features+1)

    def fit(self, x, y, weights=None, x_to_predict=None, sigma=0.5):
        """
        Fit the model using the normal equation.
        Parameters:
            x (numpy.ndarray): input data
            y (numpy.ndarray): target data
            weights (str): type of weights to use, None for ordinary least squares, "Gaussian" for Gaussian weights
            x_to_predict (numpy.ndarray): the x value to make the prediction for
            sigma (float): standard deviation for Gaussian weights
        Returns:
            numpy.ndarray: predicted y value
        """
        m, n = x.shape
        x = np.concatenate((np.ones((m, 1)), x), axis=1)
        if weights is None: # ordinary least squares
            self.theta = np.linalg.inv(x.T @ x) @ x.T @ y  
        elif weights == "Gaussian":
            weights = np.diag(np.exp(-(x_to_predict - x[:,1])**2 / (2 * sigma**2)))
            self.theta = np.linalg.inv(x.T @ weights @ x) @ x.T @ weights @ y
    def predict(self, x):
        # m, n = x.shape
        # x = np.concatenate((np.ones(1), x))
        return np.array([1, x]) @ self.theta

class Perceptron():
    """
    Implementation of the Perceptron algorithm.
    Parameters:
        num_features (int): number of features in the data
    """
    def __init__(self, num_features):
        self.num_features = num_features
        self.theta = np.zeros(num_features+1)
    
    def h(self, x):
        z = x @ self.theta
        return (np.sign(z) + 1)/2

    def gradient(self, x, y):
        return np.mean(x * (y - self.h(x))[:,None], axis=0)

    def next_theta(self, x, y, learning_rate):
        return self.theta + learning_rate * self.gradient(x, y)  # plus because we are maximizing the likelihood
    
    def fit(self, x, y,eps=1e-9, max_iter=5000, learning_rate=1):
        """ 
        Fit the model using the Perceptron algorithm.
        Parameters:
            x (numpy.ndarray): input data
            y (numpy.ndarray): target data
            eps (float): convergence threshold
            max_iter (int): maximum number of iterations
            learning_rate (float): learning rate
        """
        m, n = x.shape
        x = np.concatenate((np.ones((m, 1)), x), axis=1)
        old_theta = self.theta
        self.theta = self.next_theta(x, y, learning_rate)
        i = 0
        while np.linalg.norm(self.theta - old_theta, 1) >= eps and i < max_iter:
            old_theta = self.theta
            self.theta = self.next_theta(x, y, learning_rate)
            i += 1
    def predict(self, x):
        m, n = x.shape
        x = np.concatenate((np.ones((m, 1)), x), axis=1)
        return self.h(x) 


class PerceptronKernel():
    """
    Implementation of the Perceptron algorithm with a Gaussian kernel.
    Parameters:
        num_features (int): number of features in the data
        sigma (float): standard deviation for the Gaussian kernel
    """
    def __init__(self, num_features, sigma):
        self.num_features = num_features
        self.sigma = sigma

    def h(self, z):
        z = z + 1e-16
        return (np.sign(z) + 1)/2.0

    def kernel(self, x1, x2):
        if len(x1.shape) == 1:
            x1 = np.array([x1])
        if len(x2.shape) == 1:
            x2 = np.array([x2])
        x1 = x1[:,None,:]
        x2 = x2[None,:,:]
        return np.exp(-np.linalg.norm(x1 - x2, axis=-1)**2 / (2 * self.sigma**2))
    
    def fit(self, x, y, learning_rate=1):
        """
        Fit the data using a stochastic gradient descent algorithm doing one pass through the data.
        Parameters:
            x (numpy.ndarray): input data
            y (numpy.ndarray): target data
            learning_rate (float): learning rate
        """
        self.x_train = x
        self.y_train = y
        m, n = x.shape
        self.beta = np.zeros(m)
        self.beta[0] = learning_rate*y[0]
        for i in range(1, m):
            z = np.sum(self.beta[:i] * self.kernel(x[i], x[:i])[0,:])
            self.beta[i] = learning_rate*(y[i] - self.h(z))

    def predict(self, x):
        return self.h(np.sum(self.beta[:,None] * self.kernel(self.x_train, x), axis=0))
