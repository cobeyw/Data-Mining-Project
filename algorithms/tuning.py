from linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as LinReg
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import rmse, r_squared, ZNormalizer
from matplotlib import pyplot as plt
import os

if __name__ == "__main__":
    x_cols = ["mo","dy","mag","len","wid","area","seconds","slon","elon",
          "slat","elat","abs_dlon","abs_dlat","max_gust","min_gust","sd_gust",
          "mean_gust","median_gust"]
    target = "inj"

    df = pd.read_csv("data\\tornado_wind_data.csv")

    # preprocessing stuff, drop mag == -9 rows, replace gust NaNs with 0
    na_cols = ["max_gust","min_gust","mean_gust","sd_gust","median_gust"]
    df = df[df.mag != -9]
    df[na_cols] = df[na_cols].fillna(0)

    # looking at the spread of injuries
    no_inj = df[df.inj == 0].shape[0]
    some_inj = df[df.inj != 0].shape[0]
    nrow = df.shape[0]
    no_inj_pct = 100.0*(no_inj/nrow)
    no_inj_ideal = some_inj
    need_del = no_inj - no_inj_ideal
    print(f"{no_inj} rows w/o inj, {some_inj} rows with inj. {no_inj_pct}% of data w/o inj")
    print(f"Consider deleting {need_del} no inj rows")
    print(f"{nrow} rows before")
    no_inj_idx = df.index[df.inj == 0].tolist()
    del_idx = no_inj_idx[:need_del]
    correct_len = len(del_idx) == need_del
    print(f"Correct del length? {correct_len}")
    df.drop(del_idx, inplace=True)
    no_inj = df[df.inj == 0].shape[0]
    nrow = df.shape[0]
    no_inj_pct = 100.0*(no_inj/nrow)
    print(f"{nrow} rows after")
    print(f"{no_inj} rows w/o inj, {no_inj_pct}% of data")

    X = df[x_cols]
    znorm = ZNormalizer()
    X = znorm.fit_transform(X)
    Y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    lm = LinearRegression(l_rate=0.00001, n_iter=1000000)
    #skl_lm = LinReg()

    # fit model
    lm.fit(x_train, y_train)
    #skl_lm.fit(x_train, y_train)

    # predict
    y_train_pred = lm.predict(x_train)
    y_pred = lm.predict(x_test)
    #skl_pred = skl_lm.predict(x_test)

    obs = np.arange(y_test.shape[0])
    mets = f"RMSE = {rmse(y_test, y_pred):.4f}, R^2 = {r_squared(y_test, y_pred):.4f}"
    plt.title(mets)
    plt.scatter(obs, y_test, color="blue", label="Ground Truth")
    plt.plot(obs, y_pred, color="red", label="predicted")
    plt.xlim((0,100))
    #plt.scatter(obs, sorted(skl_pred), color="green", label="sklearn model")
    plt.legend()
    plt.show()