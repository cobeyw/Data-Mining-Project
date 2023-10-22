import numpy as np

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