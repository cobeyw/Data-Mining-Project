from linear_regression import LinearRegression, lin_reg_cv
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import rmse, r_squared, ZNormalizer, encode_states, encode_loss, adjust_dollars
from matplotlib import pyplot as plt
import pickle

# main data csv path
CSV_PATH = "data\\tornado_wind_data.csv"

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
    # get only full track data
    df = df[(df["ns"] == 1) & (df["sg"] == 1) & (df["sn"] == 1)]
    # preprocessing stuff, drop mag == -9 rows, replace gust NaNs with 0
    na_cols = ["max_gust","min_gust","mean_gust","sd_gust","median_gust"]
    df = df[df.mag != -9]
    df[na_cols] = df[na_cols].fillna(0)
    print("Adding feature interaction columns")
    # adding interaction features
    df["cas"] = df["inj"] + df["fat"]           
    df["latlong_area"] = df["abs_dlat"] * df["abs_dlon"]
    df["work_cap"] = df["len"] * (df["mag"] + 1.0)
    df["scale_vol"] = df["area"] * (df["mag"] + 1.0)
    print("Encoding states")                    
    df = encode_states(df)                      # encoding state column such tha it becomes the state tornado rank
    print("Encoding loss")
    df = encode_loss(df)                        # encode loss column as per NOAA instructions (see function for details)
    print("Adjusting for inflation")
    df = adjust_dollars(df, ["loss","closs"])   # adjust money for inflation so that it's all in current year dollars
    df["dmg"] = df["loss"] + df["closs"]        # damages target column
    # remove outliers based on user-provided outlier cutoff
    nrow_before = df.shape[0]
    if outlier_cutoff is not None:
        df = df[(df[target] >= outlier_cutoff) & (df[target] < 1e6)]
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
    tg_zero = df[df[target] == 0].shape[0]                          # number of rows with zero-valued target variable
    tg_nzero = df[df[target] != 0].shape[0]                         # number of rows with non-zero target variable
    nrow = df.shape[0]                                              # total rows in df
    tg_zero_pct = 100.0*(tg_zero/nrow)                              # percent of zero-valued target
    tg_zero_ideal = int(tg_nzero*ratio)                             # user-select percent of zero-valued target
    n_del = tg_zero - tg_zero_ideal                                 # how many rows we need to delete to meet user specs
    print(f"{tg_zero} zero-valued rows, {tg_nzero} nonzero. {tg_zero_pct}% of data zero-valued")
    print(f"{nrow} rows before")
    tg_zero_idx = df.index[df[target] == 0].tolist()                # get the indexes of zero-valued targets
    del_idx = tg_zero_idx[:n_del]                                   # decided which indexes to delete
    correct_len = len(del_idx) == n_del                             # check if we have the correct amount for deletion
    print(f"Correct del length? {correct_len}")
    df.drop(del_idx, inplace=True)                                  # drop the calculated indexes
    tg_zero = df[df[target] == 0].shape[0]                          # check new distribution of zero-valued targets
    nrow = df.shape[0]
    tg_zero_pct = 100.0*(tg_zero/nrow)                              # new pct of zero-valued targets
    print(f"{nrow} rows after")
    print(f"{tg_zero} zero-valued rows, {tg_zero_pct}% of data")
    return df

if __name__ == "__main__":
    """
    get the user inputs in order to use CV to train various
    number of iterations and learning rates for the LinearRegression
    model in order to find the best-performing combo.
    """
    #x_cols = ["state_rank","mag","len","wid","area","mean_gust","max_gust"]
    #x_cols = ["area", "mag", "wid", "mean_gust", "max_gust", "len", "min_gust","yr"]
    #x_cols = ["scale_vol", "area", "work_cap", "len", "mag"]
    x_cols = ["scale_vol", "area", "work_cap", "len", "mag", "wid"]

    target = input("Choose target col: ")
    outlier_cutoff = float(input("Choose outlier min cutoff (-1 for None): ")) # dmg = 1e6, cas = -1
    outlier_cutoff = None if outlier_cutoff < 0 else outlier_cutoff

    df = process_csv(CSV_PATH, target, outlier_cutoff)
    # do_not_include = ["inj","fat","loss","closs","cas","dmg","cas_ratio","st","date","time",
    #                    "ns","sn","sg","om","f1","f2","f3","f4","fc","stf","stn"]
    # x_cols = [c for c in df.columns.values if c not in do_not_include]
    # print(x_cols)

    # looking at the spread of target so we can shape the training data
    # to have about an equal number of tornados with and without target 
    # values = 0
    ratio = float(input("Choose zero/nonzero target values ratio (-1 to keep intact): ")) # dmg = 1.0, cas = 1.0
    if ratio != -1:
        df = fix_target_spread(df, target, ratio)

    X = df[x_cols]
    plt.title(target)
    plt.hist(df[target])
    plt.show()

    try:
        with open("models\\regr_data_norm.pkl", "rb") as f:
            znorm = pickle.load(f)
            X = znorm.transform(X).to_numpy(dtype="float32")
    except FileNotFoundError:
        znorm = ZNormalizer()
        X = znorm.fit_transform(X).to_numpy(dtype="float32")
        with open("models\\regr_data_norm.pkl", "wb") as f:
            pickle.dump(znorm, f)
    Y = df[target].to_numpy(dtype="float32")

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    run_cv = bool(int(input("Run cross-validation to find best params (1=Yes,0=No)? ")))

    if run_cv:
        n_params = int(input("How many params to try? "))
        lrs = np.linspace(1e-5, 1e-2, n_params)
        nis = np.linspace(50000, 200000, n_params)
        params, cv_r2, cv_rmse = lin_reg_cv(X, Y, lrs, nis)
        lr, ni = params
        print(f"Best: lr = {lr}, n_iter = {ni}")
        print(f"CV R^2 = {cv_r2: .4f}, CV RMSE = {cv_rmse: .4f}")
        lm = LinearRegression(l_rate=lr, n_iter=int(ni))
        lm.cv_r2 = cv_r2
        lm.cv_rmse = cv_rmse
    else:
        lr, ni = (0.005005, 125000)
        lm = LinearRegression(l_rate=lr, n_iter=int(ni))

    # fit model
    print(f"Fitting best model for {target}")
    lm.fit(x_train, y_train)
    lm.x_cols = x_cols
    lm.target = target
    lm.outlier_cutoff = outlier_cutoff
    lm.target_ratio = ratio
    print(lm.get_coeff())

    # predict
    print("Testing model")
    y_train_pred = lm.predict(x_train)
    y_pred = lm.predict(x_test)
    train_err = rmse(y_train, y_train_pred)
    train_r2 = r_squared(y_train, y_train_pred)
    print(f"Model performance: train RMSE = {train_err}, train R^2 = {train_r2}")

    # plot test performance
    obs = np.arange(y_test.shape[0])
    step = 20
    mets = (
            f"Test/train RMSE = {rmse(y_test, y_pred):.4f}/{train_err:.4f}\n"
            f"Test/train R^2 = {r_squared(y_test, y_pred):.4f}/{train_r2:.4f}"
    )
    outs = sorted(list(zip(y_pred, y_test)), key=lambda x: x[1])
    y_pred = [x[0] for x in outs]
    y_test = [x[1] for x in outs]

    plt.title(mets)
    plt.scatter(obs[::step], y_test[::step], color="blue", label="Ground Truth")
    plt.plot(obs[::step], y_pred[::step], color="red", label="predicted")
    plt.ylabel(target)
    plt.legend()
    plt.show()

    # save the model if needed
    save_model = bool(int(input("Save model (1=Yes, 0=No)? ")))
    if save_model:
        lr_str = str(lr).replace(".", "-")
        r_str = str(ratio).replace(".","-")
        name = f"lr_{lr_str}_ni_{int(ni)}_r_{r_str}"
        model_file = "models\\" + name + f"_{target}_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(lm, f)