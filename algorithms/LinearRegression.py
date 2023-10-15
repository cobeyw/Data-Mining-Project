""" Implements a class that performs Ordinary Least Squares (OLS) Linear Regression.
    NOTE: this can only perform LR for a single target variable.
"""
import numpy as np 
  
# LinearRegression class
class LinearRegression() : 
    def __init__(self, l_rate, n_iter) : 
        self.l_rate = l_rate 
        self.n_iter = n_iter 
          
    """Adjusts weights to fit Y to some input data X.
        such that y_pred = sum(Wi * Xi) + b.

        Args:
            X (np.array): numpy array of shape (n_observations, n_features).
            Y (np.array): numpy array with true values of shape (n_observations,1).
            weights (np.array): numpy with starting values for weights,
                of shape (n_features,1), defaults to zeros.
            bias (float): bias value for loss function, defaults to 0
    """         
    def fit(self, X, Y, weights=None, bias=0) : 
        # get the number of observations and features
        self.n_obs, self.n_feat = X.shape 
          
        # get the initial weights
        self.weights = np.zeros(self.n_obs) if weights is None else weights

        # record bias and X/Y arrays
        self.bias = bias
        self.X = X 
        self.Y = Y 
        
        # run iterations updating the weights
        for i in range(self.n_iter):
            self.__update_weights() 

    """Updates the weights based and bias based on
        gradients of the current weights. Meant for
        internal class use only.
    """ 
    def __update_weights(self) : 
        # get predictions for this iteration
        Y_pred = self.predict(self.X) 
          
        # weights and bias gradients
        d_weights = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.n_obs 
        d_bias = - 2 * np.sum(self.Y - Y_pred) / self.n_obs 
          
        # update weights and bias
        self. weights = self.weights - (self.l_rate * d_weights)
        self.bias = self.bias - (self.l_rate * d_bias) 

      
    """Predicts the target value for a set of inputs X,
    based on y_pred = x*weights + bias.

    Args:
        X (np.array): numpy array must have shape(m, n) where
            n == self.n_feat.
    
    Returns:
        np.array: array of shape(m, 1) with predicted values,
            None if shape is incorrect. 
    """
    def predict(self, X) : 
        if X.shape[1] != self.n_feat:
            print(f"Expected array of shape (m, {self.n_feats}), got array of shape {X.shape}")
            return None
        return X.dot(self.weights) + self.bias 