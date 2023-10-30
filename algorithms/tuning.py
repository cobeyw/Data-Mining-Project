from linear_regression import LinearRegression, lin_reg_cv
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import rmse, r_squared, ZNormalizer
from matplotlib import pyplot as plt
import pickle

"""
Preprocesses csv into cleaned df by deleting
mag == -9 rows, removing outliers, and filling
in missing gust data with 0.

Args:
      csv_path (str): path to dataframe as csv file
      target (str): column name for target variable
      outlier_cutoff (float): the highest value target
            can take before being considered an outlier
            (default is None meaning don't drop outliers)
Returns:
      pd.DataFrame: cleaned dataframe from csv
"""
def process_csv(csv_path, target, outlier_cutoff):
    df = pd.read_csv(csv_path)

    # preprocessing stuff, drop mag == -9 rows, replace gust NaNs with 0
    na_cols = ["max_gust","min_gust","mean_gust","sd_gust","median_gust"]
    df = df[df.mag != -9]
    df[na_cols] = df[na_cols].fillna(0)
    nrow_before = df.shape[0]
    if outlier_cutoff is not None:
        df = df[df[target] <= outlier_cutoff]
    nrow_after = df.shape[0]
    row_rem = nrow_before - nrow_after
    rem_pct = 100.0 * (row_rem / nrow_before)
    print(f"Removed {row_rem} rows with outliers, {rem_pct:.2f}% of original data.")
    return df

"""
Fixes target spread in dataframe so it has the
desired ratio of zero-valued target data to
non-zero-valued target data. Ex.: if ratio = 1.0
then there will be the same amount of zero and non-
zero target data.

Args:
      df (pd.DataFrame): input dataframe, must have target column
      target (str): name of target column in df
      ratio (float): z / n where z, n are the number of zero and
            nonzero target valued rows in the output dataframe
            (must not be higher than what's already in df)
Returns:
      pd.DataFrame: dataframe with the corrected ratio
"""
def fix_target_spread(df, target, ratio):
    tg_zero = df[df[target] == 0].shape[0]
    tg_nzero = df[df[target] != 0].shape[0]
    nrow = df.shape[0]
    tg_zero_pct = 100.0*(tg_zero/nrow)
    tg_zero_ideal = int(tg_nzero*ratio)
    n_del = tg_zero - tg_zero_ideal
    print(f"{tg_zero} zero-valued rows, {tg_nzero} nonzero. {tg_zero_pct}% of data zero-valued")
    print(f"{nrow} rows before")
    tg_zero_idx = df.index[df[target] == 0].tolist()
    del_idx = tg_zero_idx[:n_del]
    correct_len = len(del_idx) == n_del
    print(f"Correct del length? {correct_len}")
    df.drop(del_idx, inplace=True)
    tg_zero = df[df[target] == 0].shape[0]
    nrow = df.shape[0]
    tg_zero_pct = 100.0*(tg_zero/nrow)
    print(f"{nrow} rows after")
    print(f"{tg_zero} zero-valued rows, {tg_zero_pct}% of data")
    return df

if __name__ == "__main__":
    x_cols = ["mo","dy","mag","len","wid","area","seconds","slon","elon",
          "slat","elat","abs_dlon","abs_dlat","max_gust","min_gust","sd_gust",
          "mean_gust","median_gust"]

    csv_path = "data\\tornado_wind_data.csv"
    target = input("Choose target col: ") #
    outlier_cutoff = int(input("Choose outlier max cutoff (-1 for None): ")) #inj = 175, loss = 2e7
    outlier_cutoff = None if outlier_cutoff < 0 else outlier_cutoff

    df = process_csv(csv_path, target, outlier_cutoff)

    # looking at the spread of target so we can shape the training data
    # to have about an equal number of tornados with and without target 
    # values = 0
    ratio = float(input("Choose zero/nonzero target values ratio: "))
    df = fix_target_spread(df, target, ratio)

    X = df[x_cols]
    znorm = ZNormalizer()
    X = znorm.fit_transform(X)
    Y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    lrs = np.linspace(1e-5, 1e-3, 4)
    nis = np.linspace(50000, 200000, 4)

    run_cv = bool(int(input("Run cross-validation to find best params (1=Yes,0=No)? ")))

    if run_cv:
        lr, ni = lin_reg_cv(X, Y, lrs, nis)
        print(f"Best: lr = {lr}, n_iter = {ni}")
    else:
        lr, ni = (0.00034, 100000)
    
    # fit model
    print(f"Fitting best model for {target}")
    lm = LinearRegression(l_rate=lr, n_iter=int(ni))
    lm.fit(x_train, y_train)

    # predict
    print("Testing model")
    y_train_pred = lm.predict(x_train)
    y_pred = lm.predict(x_test)
    train_err = rmse(y_train, y_train_pred)
    train_r2 = r_squared(y_train, y_train_pred)
    lm.train_r2 = train_r2
    lm.train_rmse = train_err
    print(f"Model performance: train RMSE = {train_err}, train R^2 = {train_r2}")

    obs = np.arange(y_test.shape[0])
    mets = f"RMSE = {rmse(y_test, y_pred):.4f}, R^2 = {r_squared(y_test, y_pred):.4f}"
    plt.title(mets)
    plt.scatter(obs, sorted(y_test), color="blue", label="Ground Truth")
    plt.plot(obs, sorted(y_pred), color="red", label="predicted")
    #plt.xlim((0,100))
    plt.legend()
    plt.show()

    save_model = bool(int(input("Save model/normalizer (1=Yes, 0=No)? ")))
    if save_model:
        lr_str = str(lr).replace(".", "-")
        r_str = str(ratio).replace(".","-")
        name = f"lr_{lr_str}_ni_{ni}_r_{r_str}"
        model_file = "models\\" + name + f"_{target}_model.pkl"
        norm_file = "models\\" + name + f"_{target}_norm.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(lm, f)
        with open(norm_file, "wb") as f:
            pickle.dump(znorm, f)