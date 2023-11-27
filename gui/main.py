import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
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
INJ_MODEL_PATH = "models\\lr_0-000505_ni_200000_r_0-0_inj_model.pkl"
INJ_ERR_MODEL_PATH = "models\\lr_1e-05_ni_50000_r_inj_abserr_model.pkl"
FAT_MODEL_PATH = "models\\lr_0-001_ni_200000_r_0-1_fat_model.pkl"
FAT_ERR_MODEL_PATH = "models\\lr_1e-05_ni_200000_r_fat_abserr_model.pkl"
LOSS_MODEL_PATH = "models\\lr_0-001_ni_125000_r_0-0_loss_model.pkl"
#LOSS_ERR_MODEL_PATH = "models\\lr_0-00034_ni_150000_r_loss_abserr_model.pkl"
CLOSS_MODEL_PATH = "models\\lr_0-001_ni_50000_r_0-0_closs_model.pkl"
#CLOSS_ERR_MODEL_PATH = "models\\lr_0-00067_ni_200000_r_closs_abserr_model.pkl"
NORM_PATH = "models\\data_norm.pkl"
with open(RANKS_PATH, "rb") as f:
    RANKS = pickle.load(f)
with open(INJ_MODEL_PATH, "rb") as f:
    INJ_MODEL = pickle.load(f)
with open(FAT_MODEL_PATH,"rb") as f:
    FAT_MODEL = pickle.load(f)
with open(LOSS_MODEL_PATH, "rb") as f:
    LOSS_MODEL = pickle.load(f)
with open(CLOSS_MODEL_PATH, "rb") as f:
    CLOSS_MODEL = pickle.load(f)
with open(NORM_PATH, "rb") as f:
    DATA_NORM = pickle.load(f)
with open(INJ_ERR_MODEL_PATH, "rb") as f:
    INJ_ERR_MODEL = pickle.load(f)
with open(FAT_ERR_MODEL_PATH, "rb") as f:
    FAT_ERR_MODEL = pickle.load(f)
#with open(LOSS_ERR_MODEL_PATH, "rb") as f:
#    LOSS_ERR_MODEL = pickle.load(f)
#with open(CLOSS_ERR_MODEL_PATH, "rb") as f:
#    CLOSS_ERR_MODEL = pickle.load(f)

# GLOBALS
inj_pred = None
inj_err = None
fat_pred = None
far_err = None
loss_pred = None
loss_err = None
closs_pred = None
closs_err = None

# prediction button response
def _run_prediction_reg(user_inputs: dict, outputs: dict):
    in_cols = ["mo","dy","state_rank","mag","len","wid","area","seconds","slon","elon",
          "slat","elat","abs_dlon","abs_dlat","max_gust","min_gust","sd_gust",
          "mean_gust","median_gust"]
    # if we have a csv loaded
    if DATA_LOADED_REGRESSION:
        _ = messagebox.showinfo("CSV", "Predicting...")
        return
    # no csv loaded
    else:
        state = user_inputs["State"].get()
        try:
            month = int(user_inputs["Month"].get("1.0","end-1c"))
            day = int(user_inputs["Day"].get("1.0","end-1c"))
            mag = float(user_inputs["Magnitude"].get("1.0","end-1c"))
            tlen = float(user_inputs["Track Legnth (mi)"].get("1.0","end-1c"))
            wid = float(user_inputs["Storm Width (yd)"].get("1.0","end-1c"))
            hours = float(user_inputs["Mil Time (i.e. 0900)"].get("1.0","end-1c"))
            slon = float(user_inputs["Start Longitude"].get("1.0","end-1c"))
            elon = float(user_inputs["End Longitude"].get("1.0","end-1c"))
            slat = float(user_inputs["Start Latitude"].get("1.0","end-1c"))
            elat = float(user_inputs["End Latitude"].get("1.0","end-1c"))
        except ValueError as val_err:
            _ = messagebox.showinfo("Value Error", str(val_err))
            return
        
        area = (MI_TO_YD*tlen) * wid
        state_rank = RANKS[state]
        secs = hours * HR_TO_SEC
        abs_dlon = abs(slon - elon)
        abs_dlat = abs(slat - elat)
        gust = get_wind_data(state, month, day)
        x = [month, day, state_rank, mag, tlen, wid, area, secs, slon,
                        elon, slat, elat, abs_dlon, abs_dlat, gust["max"], gust["min"],
                        gust["stdev"], gust["mean"], gust["median"]]
        xdf = pd.DataFrame(columns=in_cols)
        xdf.loc[0] = x
        x_norm = DATA_NORM.transform(xdf).to_numpy()
        inj_pred = INJ_MODEL.predict(x_norm)[0]
        fat_pred = FAT_MODEL.predict(x_norm)[0]
        loss_pred = LOSS_MODEL.predict(x_norm)[0]
        if loss_pred < float(0):
            loss_pred = 0.0
        closs_pred = CLOSS_MODEL.predict(x_norm)[0]
        if closs_pred < float(0):
            closs_pred = 0.0
        x_inj_err = np.array([mag, inj_pred]).reshape((1,2))
        x_fat_err = np.array([mag, fat_pred]).reshape((1,2))
        #x_loss_err = np.array([mag, loss_pred]).reshape((1,2))
        #x_closs_err = np.array([mag, closs_pred]).reshape((1,2))
        inj_err = INJ_ERR_MODEL.predict(x_inj_err)[0]
        fat_err = FAT_ERR_MODEL.predict(x_fat_err)[0]
        #loss_err = LOSS_ERR_MODEL.predict(x_loss_err)[0]
        #closs_err = CLOSS_ERR_MODEL.predict(x_closs_err)[0]
        # adjust loss and crop loss as they are output on a log scale
        #loss_pred = 10**loss_pred
        #closs_pred = 10**closs_pred
        #loss_err = 10**loss_err
        #closs_err = 10**closs_err
        inj_text = f"{int(inj_pred)} +/- {inj_err:.2f}"
        fat_text = f"{int(fat_pred)} +/- {fat_err:.2f}"
        loss_text = f"${loss_pred:.2f}"
        closs_text = f"${closs_pred:.2f}"
        outputs["Injuries"].config(text=inj_text)
        outputs["Fatalities"].config(text=fat_text)
        outputs["Property Loss"].config(text=loss_text)
        outputs["Crop Loss"].config(text=closs_text)
        
# load csv button response
def _load_csv_reg(pred_button):
    global DATA_LOADED_REGRESSION
    DATA_LOADED_REGRESSION = True
    pred_button.config(text="Predict CSV")

# clear csv button response
def _clear_csv_data_reg(pred_button):
    global DATA_LOADED_REGRESSION
    DATA_LOADED_REGRESSION = False
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

# sets up casualty prediction tab
def _setup_casualty_tab(cas_tab):
    # user input variables, used in loop to make widget creation cleaner
    cas_user_inputs = ["Month", "Day", "Mil Time (i.e. 0900)", "State", "Magnitude",
                       "Track Legnth (mi)", "Storm Width (yd)", "Start Latitude", 
                       "Start Longitude", "End Latitude", "End Longitude"]
    
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
        if s == "State":
            st = tk.StringVar()
            st.set("")
            d = ttk.Combobox(cas_tab, textvariable=st, values=list(RANKS.keys()))
            d.current(1)
            d.place(relx=(xpos+wid), rely=ypos, relwidth=wid, relheight=0.05)
            user_inputs[s] = d
        else:
            t = tk.Text(cas_tab)
            t.place(relx=(xpos+wid), rely=ypos, relwidth=wid, relheight=0.05)
            user_inputs[s] = t

    # prediction outputs, similar setup to user inputs but on the other side
    outputs = ["Injuries","Fatalities","Property Loss","Crop Loss"]
    output_labels = {}
    new_xpos = 0.5+xpos
    preds = tk.Label(cas_tab, text="Predictions", bg="LightGoldenrod1")
    preds.place(relx=(new_xpos+(wid/2)), rely=0.01, anchor="nw", relwidth=wid)
    for i, s in enumerate(outputs):
        l = tk.Label(cas_tab, text=s, bg="LightGoldenrod1")
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