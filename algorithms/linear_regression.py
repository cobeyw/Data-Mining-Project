""" Implements a class that performs Ordinary Least Squares (OLS) Linear Regression.
    NOTE: this can only perform LR for a single target variable.
"""
from utils import rmse, r_squared
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LinReg
from itertools import product
  
class LinearRegression() : 
    def __init__(self, l_rate, n_iter): 
        self.l_rate = l_rate 
        self.n_iter = n_iter
        self.cv_rmse = None
        self.cv_r2 = None
        self.outlier_cutoff = -1
        self.target_ratio = None
        self.x_cols = None
        self.target = None
        self.weights = None
        self.bias = None
          
    """Adjusts weights to fit Y to some input data X.
        such that y_pred = sum(Wi * Xi) + b.

        Args:
            X (np.array): numpy array of shape (n_observations, n_features)
            Y (np.array): numpy array with true values of shape (n_observations,)
            bias (float): bias value for loss function, defaults to 0
    """         
    def fit(self, X, Y, bias=0): 
        # get the number of observations and features
        self.n_obs = X.shape[0]
        self.n_feat = 0
        if(len(X.shape) < 2):
            self.n_feat = 1
        else:
            self.n_feat = X.shape[1]
          
        # get the initial weights
        self.weights = np.zeros(self.n_feat, dtype="float32")

        # record bias and X/Y arrays
        self.bias = bias
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32) 
        
        # run iterations updating the weights
        for _ in range(self.n_iter):
            self.__update_weights() 

    """Updates the weights based and bias based on
        gradients of the current weights. Meant for
        internal class use only.
    """ 
    def __update_weights(self): 
        # get predictions for this iteration
        Y_pred = self.predict(self.X)
          
        # weights and bias gradients
        d_bias = -2.0 * np.sum(self.Y - Y_pred) / self.n_obs 
        d_weights = -2.0 * (self.X.T).dot(self.Y - Y_pred) / self.n_obs 

        # update weights and bias
        self.weights = self.weights - (self.l_rate * d_weights)
        self.bias = self.bias - (self.l_rate * d_bias) 
      
    """Predicts the target value for a set of inputs X,
    based on y_pred = x*weights + bias.

    Args:
        X (np.array): numpy array must have shape(m, n) where
            n == self.n_feat.
    
    Returns:
        np.array: array of shape(m,) with predicted values,
            None if shape is incorrect. 
    """
    def predict(self, X): 
        if X.shape[1] != self.n_feat:
            print(f"Expected array of shape (m, {self.n_feat}), got array of shape {X.shape}")
            return None
        return X.dot(self.weights) + self.bias
    
    """
    Returns the coefficients for the model.
    Returns
        Dict[str, float]: coefficients as a dict
    """ 
    def get_coeff(self):
        coeff = {}
        for feat, w in zip(self.x_cols, self.weights):
            coeff[feat] = w
        return coeff

"""
Performs k-folds cross-validation using linear regression and
    picks the best combo based on R^2
Args:
    X (pd.DataFrame): input data
    Y (pd.DataFrame): target data
    l_rates (List[float]): list of learning rates to try
    n_iters (List[int]): list of number of iterations to try
    val_size (float): validation data proportion, between (0, 1),
        defaults to 0.2

Returns:
    Tuple(float, int), float, float : best learning rate and n_iter 
        combination, along with CV R^2 and RMSE
"""  
def lin_reg_cv(X, Y, l_rates, n_iters, val_size=0.2, k=3):
    params = list(product(l_rates, n_iters))
    best_r2 = 1e-9
    best_params = None
    best_rmse = None

    for lr, ni in params:
        errs = float(0)
        r2s = float(0)
        for i in range(k):
            x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=val_size)
            print(f"Fold {i+1}, params: lr = {lr}, ni = {ni}")
            lm = LinearRegression(l_rate=lr, n_iter=int(ni))
            lm.fit(x_train, y_train)
            y_pred = lm.predict(x_val)
            r2s += (r_squared(y_val, y_pred))
            errs += (rmse(y_val, y_pred))
        avg_rmse = errs / k
        avg_r2 = r2s / k
        print(f"Avg RMSE = {avg_rmse: .4f}, Avg R^2 = {avg_r2: .4f}")
        if avg_r2 > best_r2:
            best_params = (lr, ni)
            best_r2 = avg_r2
            best_rmse = avg_rmse

    return best_params, best_r2, best_rmse 

if __name__ == "__main__":
    # create sample data
    n = 100
    m = 0.5
    b = 0
    np.random.seed(123)
    noise = np.random.normal(0,1,n)
    x = np.arange(n)
    y = (x*m) + b + noise
    df = pd.DataFrame({"X":x, "Y":y})
    X = df.iloc[:,:-1].values
    Y = df.iloc[:,1].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # create models
    lm = LinearRegression(l_rate=0.0001, n_iter=1000)
    skl_lm = LinReg()

    # fit model
    lm.fit(x_train, y_train)
    skl_lm.fit(x_train, y_train)

    # predict
    y_pred = lm.predict(x_test)
    skl_pred = skl_lm.predict(x_test)

    obs = np.arange(y_pred.shape[0])
    mets = f"RMSE = {rmse(y_test, y_pred):.4f}, R^2 = {r_squared(y_test, y_pred):.4f}"
    plt.title(mets)
    plt.scatter(obs, sorted(y_test), color="blue", label="Ground Truth")
    plt.plot(obs, sorted(y_pred), color="red", label="predicted")
    plt.scatter(obs, sorted(skl_pred), color="green", label="sklearn model")
    plt.legend()
    plt.show()