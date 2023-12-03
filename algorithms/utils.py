import numpy as np
import pickle
import pandas as pd

# CONSTANTS
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
    for state, sdf in df.groupby(["st"]):                   # for each state dataset
        state = state[0]                                    # extract state code
        weight_sum = float(0)                               # initialize the weighted sum
        tot_years = sdf["yr"].max() - sdf["yr"].min() + 1   # get span of years (+1 to prevent div by 0)
        for ef in range(6):                                 # for each EF rating (0-5)
            n = sdf[sdf["mag"] == ef].shape[0]              # number of tornadoes at this EF for this state
            weight_sum += (n * ef)                          # add to weighted sum
        st_rank = weight_sum / tot_years                    # calc state rank
        ranks[state] = st_rank                              # record state rank
    df["state_rank"] = df["st"].replace(ranks)              # replace/rename column
    with open("data\\state_ranks.pkl", "wb") as f:          # update the file containing state ranks
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
        if row["loss"] == float(0):                 # zero stays as zero
            continue
        elif row["yr"] <= 1996:                     # pre-1996 loss is on a log scale of sorts
            loss_code = row["loss"]+1               # undo log (+1 since we are skipping zero)
            # this equation puts the value at the center point 
            # of the range given by NOAA. They set each number
            # from 1-8 as representative of a range as opposed 
            # to a single value
            row["loss"] = 2.5 * (10 ** loss_code)   
        elif row["yr"] > 1996:                      # post 1996 loss is in millions of dollars
            row["loss"] = row["loss"] * 1e6
    df["closs"] = df["closs"] * 1e6                 # crop loss is always in millions of dollars
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
    curr_year = CPI["Year"].max()                                   # get the current year
    curr_cpi = CPI[CPI["Year"] == curr_year]["CPI"].values[0]       # get the current year CPI
    for _, row in df.iterrows():                                    # for each row in dataframe
        year_cpi =CPI[CPI["Year"] == row["yr"]]["CPI"].values[0]    # get the CPI for this year's row
        for col in dollar_cols:                                     # for each dollar column
            adjustment = float(curr_cpi)/year_cpi                   # calculate adjustment based on CPI ratio
            row[col] = row[col] * adjustment                        # adjust the dollar amount
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

    """
    Fit the normalizer to some data by using z score

    Args:
        X (pd.DataFrame): df of shape n_sample, n_features
    
    Returns:
        ZNormalizer: the fitted object
    """
    def fit(self, X, _=None):
        nrow = float(X.shape[0])        # get number of rows
        for col in X.columns.values:    # for each column
            mean = (sum(X[col]) / nrow) # get the mean value for this column
            std = X[col].std()          # get st dev for this column
            self.__mean[col] = mean     # record these metrics
            self.__stdev[col] = std
        self.__is_fit = True            # record the data is fit
        return self
    
    """
    Transforms the data using z score normalization after fit

    Args:
        X (pd.DataFrame): df of shape n_sample, n_features

    Returns:
        pd.DataFrame: transformed dataframe
    """
    def transform(self, X):
        if not self.__is_fit:                       # fit if it is not fit yet
            self.fit(X)
        for col in X.columns.values:                # for each column
            col_vals = X[col].values                # get the values
            mean = self.__mean[col]                 # get the metrics
            std = self.__stdev[col]
            norm_col = col_vals                     # make a copy
            if std == float(0):                     # special case: st dev is 0 so mean must be 1
                norm_col.fill(1)
            else:                                   # normal case
                norm_col = (col_vals - mean) / std  # get z score
            X.loc[:,col] = norm_col                 # replace values
        return X
    
    # calls fit followed by transform
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    """
    Inverse of transform to get original values back

    Args:
        X (pd.DataFrame): pre-transformed data

    Returns:
        pd.DataFrame: data in original units
    """
    def inverse_transform(self, X):
        if not self.__is_fit:
            return None
        
        for col in X.columns.values:                    # for each column
            mean = self.__mean[col]                     # get metrics
            std = self.__stdev[col]
            col_vals = X[col].values                    # make copy
            if std == float(0):                         # special case: st dev is 0 so fill with mean
                X.loc[:,col] = col_vals.fill(mean)      
            else:                                       # normal case: undo z score
                X.loc[:,col] = (col_vals * std) + mean
        return X