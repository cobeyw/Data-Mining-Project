from tuning import process_csv, fix_target_spread, CSV_PATH
from linear_regression import LinearRegression, lin_reg_cv
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from utils import rmse, r_squared
from matplotlib import pyplot as plt
import pandas as pd

def load_model_and_norm(m_path, n_path=None):
    model = None
    norm = None
    with open(m_path, "rb") as f:
        model = pickle.load(f)
    if n_path is not None:
        with open(n_path, "rb") as f:
            norm = pickle.load(f)
    return model, norm

if __name__ == "__main__":
    m_path = input("Enter model file path: ")
    n_path =  input("Enter znorm file path (blank if not needed): ")
    if n_path == "":
        n_path = None
    model, norm = load_model_and_norm(m_path, n_path)
    df = process_csv(CSV_PATH, model.target, model.outlier_cutoff)
    df = fix_target_spread(df, model.target, model.target_ratio)
    X1 = df[model.x_cols]
    if norm is not None:
        X1 = norm.transform(X1)
    Y1 = df[model.target]
    
    Ypred = model.predict(X1)
    df["pred"] = Ypred
    df["abs_err"] = abs(Ypred - Y1)

    cols = ["mag", "pred", model.target]
    x_cols = ["mag", "pred"]
    target = "abs_err"
    
    lrs = np.linspace(1e-5, 1e-3, 4)
    nis = np.linspace(50000, 200000, 4)

    run_cv = bool(int(input("Run cross-validation to find best params (1=Yes,0=No)? ")))

    if run_cv:
        lr, ni = lin_reg_cv(df[x_cols], df[target], lrs, nis)
        print(f"Best: lr = {lr}, n_iter = {ni}")
    else:
        lr, ni = (1e-3, 200000)

    X = df[cols]
    Y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    ground_truth = x_test[model.target]
    x_train = x_train[x_cols]
    x_test = x_test[x_cols]

    lm_err = LinearRegression(l_rate=lr, n_iter=int(ni))
    lm_err.fit(x_train, y_train)
    lm_err.x_cols = x_cols
    lm_err.target = target

    y_train_pred = lm_err.predict(x_train[x_cols])
    y_pred = lm_err.predict(x_test)
    train_err = rmse(y_train, y_train_pred)
    train_r2 = r_squared(y_train, y_train_pred)
    lm_err.train_r2 = train_r2
    lm_err.train_rmse = train_err
    print(f"Model performance: train RMSE = {train_err}, train R^2 = {train_r2}")
    upper_bound = x_test["pred"] + y_pred
    lower_bound = x_test["pred"] - y_pred

    obs = np.arange(y_test.shape[0])
    mets = f"RMSE = {rmse(y_test, y_pred):.4f}, R^2 = {r_squared(y_test, y_pred):.4f}"
    plt.title(mets)
    plt.plot(obs, ground_truth, color="black", label="Ground Truth")
    plt.scatter(obs, x_test["pred"], color="green", label="predicted")
    plt.scatter(obs, upper_bound, color="red", marker="^", label="Upper Bound Error")
    plt.scatter(obs, lower_bound, color="blue", marker="v", label="Lower Bound Error")
    plt.ylabel(model.target)
    plt.xlim(0, 30)
    plt.legend()
    plt.show()

    save_model = bool(int(input("Save model/normalizer (1=Yes, 0=No)? ")))
    if save_model:
        lr_str = str(lr).replace(".", "-")
        name = f"lr_{lr_str}_ni_{int(ni)}_r_{model.target}"
        model_file = "models\\" + name + f"_abserr_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(lm_err, f)