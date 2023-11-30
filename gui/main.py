import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter import simpledialog
from idlelib.tooltip import Hovertip
import pickle
import pandas as pd
import numpy as np
import sys
sys.path.append("algorithms")

# CONSTANTS
MI_TO_YD = 1760.0
HR_TO_SEC = 3600
CSV_PATH = "data\\tornado_wind_data.csv"
WIND_CSV_PATH = "data\\wind.csv"
WIND = None
DATA_LOADED_REGRESSION = False
RANKS_PATH = "data\\state_ranks.pkl"
CAS_MODEL_PATH = "models\\lr_0-005005_ni_200000_r_1-0_cas_model.pkl"
DMG_MODEL_PATH = "models\\lr_0-01_ni_200000_r_0-5_dmg_model.pkl"
NORM_PATH = "models\\data_norm.pkl"
with open(RANKS_PATH, "rb") as f:
    RANKS = pickle.load(f)
with open(NORM_PATH, "rb") as f:
    DATA_NORM = pickle.load(f)
with open(CAS_MODEL_PATH, "rb") as f:
    CAS_MODEL = pickle.load(f)
with open(DMG_MODEL_PATH, "rb") as f:
    DMG_MODEL = pickle.load(f)

# GLOBALS
x_in = None

# prediction button response
def _run_prediction_reg(user_inputs: dict, outputs: dict):
    global x_in
    in_cols = ["state_rank","mag","len","wid","area","mean_gust","max_gust"]
    # if we have a csv loaded
    if DATA_LOADED_REGRESSION:
        x_in_og = x_in.copy()
        x_in["area"] = (MI_TO_YD*x_in["len"]) * x_in["wid"]
        x_in = get_wind_data_df(x_in)
        x_in = x_in.replace({"st": RANKS})
        x_in = x_in.rename(columns={"st": "state_rank"})
        x_in = x_in[in_cols]
        print(x_in)
        x_norm = DATA_NORM.transform(x_in).to_numpy()
        cas_pred = CAS_MODEL.predict(x_norm)
        dmg_pred = DMG_MODEL.predict(x_norm)
        x_in_og["cas"] = cas_pred
        x_in_og["dmg"] = dmg_pred
        x_in_og.to_csv("predictions.csv")
    # no csv loaded
    else:
        state = user_inputs["State"].get()
        try:
            month = int(user_inputs["Month"].get())
            day = int(user_inputs["Day"].get())
            mag = float(user_inputs["Magnitude"].get())
            tlen = float(user_inputs["Track Length (mi)"].get())
            wid = float(user_inputs["Storm Width (yd)"].get())
        except ValueError:
            _ = messagebox.showinfo("Value Error", 
                                    "Invalid input for prediction")
            return
        
        area = (MI_TO_YD*tlen) * wid
        state_rank = RANKS[state]
        gust = get_wind_data(state, month, day)
        x = [state_rank,mag,tlen,wid,area,gust["mean"],gust["max"]]
        xdf = pd.DataFrame(columns=in_cols)
        xdf.loc[0] = x
        x_norm = DATA_NORM.transform(xdf).to_numpy()
        cas_pred = CAS_MODEL.predict(x_norm)[0]
        dmg_pred = DMG_MODEL.predict(x_norm)[0]
        cas_pred = cas_pred if cas_pred > float(0) else 0
        dmg_pred = dmg_pred if dmg_pred > float(0) else 0.0
        cas_text = f"{int(cas_pred)}"
        dmg_text = f"${(dmg_pred):.2f}"
        outputs["Casualties"].config(text=cas_text)
        outputs["Damages"].config(text=dmg_text)
        
# load csv button response
def _load_csv_reg(pred_button):
    global DATA_LOADED_REGRESSION, x_in
    csv_path = simpledialog.askstring(title="",
                                  prompt="Enter CSV file path: ")
    try:
        x_in = pd.read_csv(csv_path)
    except FileNotFoundError:
        _ = messagebox.showinfo("Error", 
                                f"File {csv_path} not found")
        return
    DATA_LOADED_REGRESSION = True
    pred_button.config(text="Predict CSV")

# clear csv button response
def _clear_csv_data_reg(pred_button):
    global DATA_LOADED_REGRESSION, x_in
    DATA_LOADED_REGRESSION = False
    x_in  = None
    pred_button.config(text="Predict")

"""
Calculates wind data given a date and state, using historical
    data from NOAA. If no data matches the criteria, we simply
    use overall state averages.

Args:
    state (str): state abbreviation (e.g.: "OK")
    month (int): month number
    day (int): day of the month

Returns:
    Dict{str: float}: dicitonary containing, min/max/mean/median/stdev
        for gusts in state on month/day
"""
def get_wind_data(state, month, day):
    global WIND
    # get wind data and match to parameters
    if WIND is None:
        WIND = pd.read_csv(WIND_CSV_PATH, low_memory=False)
    data = WIND[(WIND["st"] == state) & (WIND["mo"] == month) & (WIND["dy"] == day)]
    # if we have data matching these parameters
    if data.shape[0] > 0:
        mu = data["mag"].mean()
        sd = data["mag"].std()
        mx = data["mag"].max()
        mn = data["mag"].min()
        md = data["mag"].median()
        return {"min": mn, "max": mx, "mean": mu, 
                "median": md, "stdev": sd}
    # if we do not have data matching
    else:
        data = WIND[WIND["st"] == state]
        # if we have any data for the state
        if data.shape[0] > 0:
            mu = data["mag"].mean()
            sd = data["mag"].std()
            mx = data["mag"].max()
            mn = data["mag"].min()
            md = data["mag"].median()
            return {"min": mn, "max": mx, "mean": mu, 
                    "median": md, "stdev": sd}
        # if we have no data for the state
        else:
            return {"min": 0, "max": 0, "mean": 0, 
                    "median": 0, "stdev": 0}

"""
Fills in wind data for the entire dataframe

Args:
    df (pd.DataFrame): dataframe with st, dy, mo cols

Return
    pd.DataFrame: df with wind data
"""
def get_wind_data_df(df):
    mingust = []
    maxgust = []
    meangust = []
    mediangust = []
    stdevgust = []
    for _, row in df.iterrows():
        gust = get_wind_data(row["st"], row["mo"], row["dy"])
        mingust.append(gust["min"])
        maxgust.append(gust["max"])
        meangust.append(gust["mean"])
        mediangust.append(gust["median"])
        stdevgust.append(gust["stdev"])
    df["min_gust"] = mingust
    df["max_gust"] = maxgust
    df["mean_gust"] = meangust
    df["median_gust"] = mediangust
    df["std_gust"] = stdevgust
    return df

# sets up casualty prediction tab
def _setup_casualty_tab(cas_tab):
    # user input variables, used in loop to make widget creation cleaner
    cas_user_inputs = ["Month", "Day", "State", "Magnitude",
                       "Track Length (mi)", "Storm Width (yd)"]
    input_lims = {"Month": [1,12,1], "Day": [1,31,1], "Magnitude": [0,5,0.5],
                  "Track Length (mi)": [0.0,1000.0,0.01],
                  "Storm Width (yd)": [0.0,10000.0,0.01]}
    tools_tips = ["Number of month, i.e. January = 1",
                  "Day of month", "State abbreviation",
                  "EF-Scale magnitude", "Length of storm track in miles",
                  "Width of storm in yards"]
    # relative y step increase for each label (label spacing)
    lstep = 1.0 / (len(cas_user_inputs)+1)
    # widget width
    wid = 0.2
    # starting x position
    xpos = 0.01
    # manual data entry label
    mde = tk.Label(cas_tab, text="Manual Data Entry", bg="navy", fg="white")
    mde.place(relx=(wid/2), rely=0.01, anchor="nw", relwidth=wid)
    # dictionary will hold user input widgets, which get returned so we
    # can capture user input later
    user_inputs = {}
    # user input labels and text boxes
    for i, s in enumerate(cas_user_inputs):
        l = tk.Label(cas_tab, text=s, bg="gray60")
        ypos = (i+1) * lstep
        l.place(relx=xpos, rely=ypos, ancho="nw", relwidth=wid, relheight=0.05)
        _ = Hovertip(l, tools_tips[i], hover_delay=500)
        if s == "State":
            st = tk.StringVar()
            st.set("")
            d = ttk.Combobox(cas_tab, textvariable=st, values=list(RANKS.keys()))
            d.current(1)
            d.place(relx=(xpos+wid), rely=ypos, relwidth=wid, relheight=0.05)
            user_inputs[s] = d
        else:
            lim = input_lims[s]
            t = tk.Spinbox(cas_tab, from_=lim[0], to=lim[1],
                           increment=lim[2])
            t.place(relx=(xpos+wid), rely=ypos, relwidth=wid, relheight=0.05)
            user_inputs[s] = t

    # prediction outputs, similar setup to user inputs but on the other side
    outputs = ["Casualties", "Damages"]
    tools_tips = ["All injuries to persons, including fatal", "Damage to crops and property, in USD"]
    output_labels = {}
    new_xpos = 0.5+xpos
    preds = tk.Label(cas_tab, text="Predictions", bg="LightGoldenrod1")
    preds.place(relx=(new_xpos+(wid/2)), rely=0.01, anchor="nw", relwidth=wid)
    for i, s in enumerate(outputs):
        l = tk.Label(cas_tab, text=s, bg="LightGoldenrod1")
        _ = Hovertip(l, tools_tips[i], hover_delay=500)
        ypos = (i+1) * lstep
        l.place(relx=new_xpos, rely=ypos, ancho="nw", relwidth=wid, relheight=0.05)
        o = tk.Label(cas_tab, text="", bg="white")
        o.place(relx=(new_xpos+wid), rely=ypos, relwidth=wid, relheight=0.05)
        output_labels[s] = o

    # buttons
    pred_button = tk.Button(cas_tab, text="Predict", 
                            command=lambda: _run_prediction_reg(user_inputs, output_labels),
                            bg="navy", fg="white", relief="raised")
    load_button = tk.Button(cas_tab, text="Load CSV", command=lambda: _load_csv_reg(pred_button),
                            bg="navy", fg="white", relief="raised")
    clear_button = tk.Button(cas_tab, text="Clear CSV Data", command=lambda: _clear_csv_data_reg(pred_button),
                            bg="tomato", relief="raised")
    load_button.place(relx=0.5, rely=0.8, relwidth=wid)
    clear_button.place(relx=0.5, rely=0.7, relwidth=wid)
    pred_button.place(relx=0.5, rely=0.9, relwidth=wid)

if __name__ == "__main__":
    # WINDOW SETUP
    window = tk.Tk()
    window.geometry("800x400")
    window.title("Tornado Data Miner")
    nb = ttk.Notebook(window)
    cas_tab = ttk.Frame(nb)
    mag_tab = ttk.Frame(nb)
    pro_tab = ttk.Frame(nb)
    nb.add(cas_tab, text="Tornado Casualty Prediction") 
    nb.add(mag_tab, text="Tornado Magnitude Classification") 
    nb.add(pro_tab, text="Tornado Profiles")
    nb.pack(expand=1, fill="both") 
    
    _setup_casualty_tab(cas_tab)

    window.mainloop()