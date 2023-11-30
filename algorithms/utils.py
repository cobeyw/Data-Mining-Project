import numpy as np
from statistics import stdev
import pickle
import pandas as pd

CPI_PATH = "data\\cpi.csv"
CPI = pd.read_csv(CPI_PATH)

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
Converts the st (state) column to numeric type, by weighing
    the average number of tornadoes by the magnitude per state.
    This number is then used as the ranking.

Args:
    df (pd.DataFrame): dataframe with st, mag and yr cols

Return
    pd.DataFrame: same df with state_rank col
"""
def encode_states(df):
    ranks = {}
    for state, sdf in df.groupby(["st"]):
        state = state[0]
        weight_sum = float(0)
        tot_years = sdf["yr"].max() - sdf["yr"].min() + 1
        for ef in range(6):
            n = sdf[sdf["mag"] == ef].shape[0]
            weight_sum += (n * ef)
        st_rank = weight_sum / tot_years
        ranks[state] = st_rank
    df["state_rank"] = df["st"].replace(ranks)
    with open("data\\state_ranks.pkl", "wb") as f:
        pickle.dump(ranks, f)
    return df

"""
The loss column is a bit complicated. Before 1996, it is
    a number [0,8] with 0 = < $50 and 8 = [$5e8,$5e9] going
    up by an order of magnitude in between. After 1996 it
    is millions of dollars. They also mention it may just
    become whole dollars in the future. This function
    attempts to decode that into something uniform. For
    crop loss, it is simply in millions of dollars.

Args:
    df (pd.DataFrame): dataframe with loss, closs and yr cols

Return
    pd.DataFrame: same df with adjusted loss/closs
"""
def encode_loss(df):
    for _, row in df.iterrows():
        if row["loss"] == float(0):
            continue
        elif row["yr"] <= 1996:
            loss_code = row["loss"]+1
            row["loss"] = 2.5 * (10 ** loss_code)
        elif row["yr"] > 1996:
            row["loss"] = row["loss"] * 1e6
    df["closs"] = df["closs"] * 1e6
    return df

"""
Adjust dollar columns in df for inflation, based Consumer Price Index
Args:
    df (pd.DataFrame): dataframe with yr col and cols listed in dollar_cols
    dollar_cols (List[str]): column names of columns that contain dollar amounts

Return
    pd.DataFrame: dataframe with adjusted dollars
"""
def adjust_dollars(df, dollar_cols):
    curr_year = CPI["Year"].max()
    curr_cpi = CPI[CPI["Year"] == curr_year]["CPI"].values[0]
    for _, row in df.iterrows():
        year_cpi =CPI[CPI["Year"] == row["yr"]]["CPI"].values[0]
        for col in dollar_cols:
            adjustment = float(curr_cpi)/year_cpi
            row[col] = row[col] * adjustment
    return df

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
            