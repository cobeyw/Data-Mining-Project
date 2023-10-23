import numpy as np
from statistics import stdev

"""
Function to calculate Root Mean Squared Error

Args:
    y_true (Any): array-like or scalar true values
    y_pred (Any): array-like or scalar predicted values

Returns:
    double: RMSE value
"""
def rmse(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

"""
Function to calculate coefficient of determination (R^2)

Args:
    y_true (Any): array-like or scalar true values
    y_pred (Any): array-like or scalar predicted values

Returns:
    double: R^2 value
"""
def r_squared(y_true, y_pred):
    rss = ((y_pred - y_true) ** 2).sum()
    tss = ((y_true - np.average(y_true)) ** 2).sum()
    rsq = 1 - (rss / tss)
    return rsq

"""
Class to transform data by performing
z-score normalization. Also has the ability
to inverse-transform
"""
class ZNormalizer:
    def __init__(self):
        self.__mean = {}
        self.__stdev = {}
        self.__is_fit = False

    def fit(self, X, _=None):
        nrow = float(X.shape[0])
        for col in X.columns.values:
            mean = (sum(X[col]) / nrow)
            std = stdev(X[col])
            self.__mean[col] = mean
            self.__stdev[col] = std
        self.__is_fit = True
        return self
    
    def transform(self, X):
        if not self.__is_fit:
            self.fit(X)
        for col in X.columns.values:
            col_vals = X[col].values
            mean = self.__mean[col]
            std = self.__stdev[col]
            norm_col = col_vals
            if std == float(0):
                norm_col.fill(1)
            else:
                norm_col = (col_vals - mean) / std
            X.loc[:,col] = norm_col
        return X
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        if not self.__is_fit:
            return None
        
        for col in X.columns.values:
            mean = self.__mean[col]
            std = self.__stdev[col]
            col_vals = X[col].values
            if std == float(0):
                X.loc[:,col] = col_vals.fill(mean)
            else:
                X.loc[:,col] = (col_vals * std) + mean
        return X
            