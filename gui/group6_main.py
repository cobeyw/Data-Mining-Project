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
import k_nearest_classifier

# CONSTANTS
MI_TO_YD = 1760.0
HR_TO_SEC = 3600
CSV_PATH = "data\\tornado_wind_data.csv"
WIND_CSV_PATH = "data\\wind.csv"
WIND = None
DATA_LOADED_REGRESSION = False
DATA_LOADED_CLASSIFICATION = False
RANKS_PATH = "data\\state_ranks.pkl"
CAS_MODEL_PATH = "models\\lr_0-01_ni_50000_r_1-0_cas_model.pkl"
DMG_MODEL_PATH = "models\\lr_0-005005_ni_50000_r_0-0_dmg_model.pkl"
REGR_NORM_PATH = "models\\regr_data_norm.pkl"
with open(RANKS_PATH, "rb") as f:
    RANKS = pickle.load(f)
with open(REGR_NORM_PATH, "rb") as f:
    REGR_DATA_NORM = pickle.load(f)
with open(CAS_MODEL_PATH, "rb") as f:
    CAS_MODEL = pickle.load(f)
with open(DMG_MODEL_PATH, "rb") as f:
    DMG_MODEL = pickle.load(f)

# GLOBALS
x_in = None

# prediction button response
def _run_prediction_reg(user_inputs: dict, outputs: dict):
    global x_in
    in_cols = ["scale_vol", "area", "work_cap", "len", "mag", "wid"]
    # if we have a csv loaded
    if DATA_LOADED_REGRESSION:
        x_in_og = x_in.copy()
        x_in["area"] = (MI_TO_YD*x_in["len"]) * x_in["wid"]
        x_in["work_cap"] = x_in["len"] * (x_in["mag"] + 1.0)
        x_in["scale_vol"] = x_in["area"] * (x_in["mag"] + 1.0)
        x_in = get_wind_data_df(x_in)
        x_in = x_in.replace({"st": RANKS})
        x_in = x_in.rename(columns={"st": "state_rank"})
        x_in = x_in[in_cols]
        print(x_in)
        x_norm = REGR_DATA_NORM.transform(x_in).to_numpy()
        cas_pred = CAS_MODEL.predict(x_norm)
        dmg_pred = DMG_MODEL.predict(x_norm)
        x_in_og["cas"] = cas_pred
        x_in_og["dmg"] = dmg_pred
        x_in_og.to_csv("predictions.csv")
    # no csv loaded
    else:
        try:
            mag = float(user_inputs["EF Magnitude"].get())
            tlen = float(user_inputs["Track Length (mi)"].get())
            wid = float(user_inputs["Storm Width (yd)"].get())
        except ValueError:
            _ = messagebox.showinfo("Value Error", 
                                    "Invalid input for prediction")
            return
        
        area = (MI_TO_YD*tlen) * wid
        work_cap = tlen * (mag + 1.0)
        scale_vol = area * (mag + 1.0)
        x = [scale_vol, area, work_cap, tlen, mag, wid]
        xdf = pd.DataFrame(columns=in_cols)
        xdf.loc[0] = x
        x_norm = REGR_DATA_NORM.transform(xdf).to_numpy()
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
    cas_user_inputs = ["EF Magnitude", "Track Length (mi)", "Storm Width (yd)"]
    input_lims = {"EF Magnitude": [0,5,0.5],
                  "Track Length (mi)": [0.0,1000.0,0.01],
                  "Storm Width (yd)": [0.0,10000.0,0.01]}
    tools_tips = ["EF-Scale magnitude (EF0 - EF5)", "Length of storm track in miles",
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

# prediction button response
# Calculates predicted EF values based on manual input or CSV
def _run_prediction_mag(user_inputs: dict, outputs: dict):
    global x_in
    df = pd.read_csv(CSV_PATH)
    classifier = k_nearest_classifier.KNearestClassifier(df)
    
    # If csv input loaded, iterate through inputs and write predictions to csv
    if DATA_LOADED_CLASSIFICATION:
        x_in_og = x_in.copy()
        # create columns for predictions
        EF0preds = [0] * len(x_in_og)
        EF1preds = [0] * len(x_in_og)
        EF2preds = [0] * len(x_in_og)
        EF3preds = [0] * len(x_in_og)
        EF4preds = [0] * len(x_in_og)
        EF5preds = [0] * len(x_in_og)
        #iterate through csv input and fill columns with predictions
        for i, row in x_in_og.iterrows():
            magnitudes = classifier.classify(row.mo, row.len, row.wid, row.seconds, row.slat, row.slon, row.max_gust)
            #count occurances of EF values to get percent
            EFcounts = [0, 0,0,0,0,0] #store occurances of each ef val
            for mag in magnitudes: EFcounts[mag] += 1
            EF0preds[i] = EFcounts[0]/len(magnitudes) * 100
            EF1preds[i] = EFcounts[1]/len(magnitudes) * 100
            EF2preds[i] = EFcounts[2]/len(magnitudes) * 100
            EF3preds[i] = EFcounts[3]/len(magnitudes) * 100
            EF4preds[i] = EFcounts[4]/len(magnitudes) * 100
            EF5preds[i] = EFcounts[5]/len(magnitudes) * 100
        #add columns and write prediction csv
        x_in_og['EF0_Prediction'] = EF0preds
        x_in_og['EF1_Prediction'] = EF1preds
        x_in_og['EF2_Prediction'] = EF2preds
        x_in_og['EF3_Prediction'] = EF3preds
        x_in_og['EF4_Prediction'] = EF4preds
        x_in_og['EF5_Prediction'] = EF5preds
        x_in_og.to_csv("predictions.csv")

    # If no csv input loaded, calculate prediction based on manual entries
    else:
        # get manual inputs
        month = float(user_inputs["Month (number)"].get())
        length = float(user_inputs["Track Length (mi)"].get())
        width = float(user_inputs["Storm Width (yd)"].get())
        time = float(user_inputs["Seconds Since Midnight"].get())
        slat = float(user_inputs["Starting Latitude"].get())
        slon = float(user_inputs["Starting Longitude"].get())
        max_gust = float(user_inputs["Maximum Gust (mph)"].get())

        # Use k nearest neighbors to find predicted magnitudes
        magnitudes = classifier.classify(month, length, width, time, slat, slon, max_gust)

        #find most common magnitude
        EFcounts = [0,0,0,0,0,0] #store occurances of each ef val
        for mag in magnitudes: EFcounts[mag] += 1

        #write predicition outputs
        outputs["EF 0 Confidence"].config(text=f"{(EFcounts[0]/len(magnitudes) * 100):.2f}%")
        outputs["EF 1 Confidence"].config(text=f"{(EFcounts[1]/len(magnitudes) * 100):.2f}%")
        outputs["EF 2 Confidence"].config(text=f"{(EFcounts[2]/len(magnitudes) * 100):.2f}%")
        outputs["EF 3 Confidence"].config(text=f"{(EFcounts[3]/len(magnitudes) * 100):.2f}%")
        outputs["EF 4 Confidence"].config(text=f"{(EFcounts[4]/len(magnitudes) * 100):.2f}%")
        outputs["EF 5 Confidence"].config(text=f"{(EFcounts[5]/len(magnitudes) * 100):.2f}%")
        
        # highlight the highest confidence
        mag_pred = np.argmax(EFcounts) # mode of magnitudes. if tie use lower EF
        if mag_pred == 0:
            outputs["EF 0 Confidence"].config(bg="yellow")
            outputs["EF 1 Confidence"].config(bg="white")
            outputs["EF 2 Confidence"].config(bg="white")
            outputs["EF 3 Confidence"].config(bg="white")
            outputs["EF 4 Confidence"].config(bg="white")
            outputs["EF 5 Confidence"].config(bg="white")
        if mag_pred == 1:
            outputs["EF 0 Confidence"].config(bg="white")
            outputs["EF 1 Confidence"].config(bg="yellow")
            outputs["EF 2 Confidence"].config(bg="white")
            outputs["EF 3 Confidence"].config(bg="white")
            outputs["EF 4 Confidence"].config(bg="white")
            outputs["EF 5 Confidence"].config(bg="white")
        if mag_pred == 2:
            outputs["EF 0 Confidence"].config(bg="white")
            outputs["EF 1 Confidence"].config(bg="white")
            outputs["EF 2 Confidence"].config(bg="yellow")
            outputs["EF 3 Confidence"].config(bg="white")
            outputs["EF 4 Confidence"].config(bg="white")
            outputs["EF 5 Confidence"].config(bg="white")    
        if mag_pred == 3:
            outputs["EF 0 Confidence"].config(bg="white")
            outputs["EF 1 Confidence"].config(bg="white")
            outputs["EF 2 Confidence"].config(bg="white")
            outputs["EF 3 Confidence"].config(bg="yellow")
            outputs["EF 4 Confidence"].config(bg="white")
            outputs["EF 5 Confidence"].config(bg="white")
        if mag_pred == 4:
            outputs["EF 0 Confidence"].config(bg="white")
            outputs["EF 1 Confidence"].config(bg="white")
            outputs["EF 2 Confidence"].config(bg="white")
            outputs["EF 3 Confidence"].config(bg="white")
            outputs["EF 4 Confidence"].config(bg="yellow")
            outputs["EF 5 Confidence"].config(bg="white")
        if mag_pred == 5:
            outputs["EF 0 Confidence"].config(bg="white")
            outputs["EF 1 Confidence"].config(bg="white")
            outputs["EF 2 Confidence"].config(bg="white")
            outputs["EF 3 Confidence"].config(bg="white")
            outputs["EF 4 Confidence"].config(bg="white")
            outputs["EF 5 Confidence"].config(bg="yellow")

# load csv button response
def _load_csv_mag(pred_button):
    global DATA_LOADED_CLASSIFICATION, x_in
    csv_path = simpledialog.askstring(title="",
                                  prompt="Enter CSV file path: ")
    try:
        x_in = pd.read_csv(csv_path)
    except FileNotFoundError:
        _ = messagebox.showinfo("Error", 
                                f"File {csv_path} not found")
        return
    DATA_LOADED_CLASSIFICATION = True
    pred_button.config(text="Predict CSV")

# clear csv button response
def _clear_csv_data_mag(pred_button):
    global DATA_LOADED_CLASSIFICATION, x_in
    DATA_LOADED_CLASSIFICATION = False
    x_in = None
    pred_button.config(text="Predict")

def _setup_mag_tab(mag_tab):
    # user input variables, used in loop to make widget creation cleaner
    mag_user_inputs = ["Month (number)", "Track Length (mi)", "Storm Width (yd)", "Seconds Since Midnight", "Starting Latitude", "Starting Longitude", "Maximum Gust (mph)"]
    input_lims = {"Month (number)": [1,12,1],
                  "Track Length (mi)": [0.0,1000.0,0.01],
                  "Storm Width (yd)": [0.0,10000.0,0.01],
                  "Seconds Since Midnight": [0.0,100000.0,0.01],
                  "Starting Latitude": [0.0,200.0,0.01],
                  "Starting Longitude": [-200,0,0.01],
                  "Maximum Gust (mph)": [0.0,500.0,0.01]}
    
    tools_tips = ["Month occurred January=1 December = 12", "Length of storm track in miles",
                  "Width of storm in yards", "Seconds passed since midnight (hours * 3600)", "Latitude of where tornado started ~-100", "Longitude of where tornado started", "Maximum gust record in mph"]
    # relative y step increase for each label (label spacing)
    lstep = 0.8 / (len(mag_user_inputs)+1)
    # widget width
    wid = 0.2
    # starting x position
    xpos = 0.01
    # manual data entry label
    mde = tk.Label(mag_tab, text="Manual Data Entry", bg="navy", fg="white")
    mde.place(relx=(wid/2), rely=0.01, anchor="nw", relwidth=wid)
    # dictionary will hold user input widgets, which get returned so we
    # can capture user input later
    user_inputs = {}
    # user input labels and text boxes
    for i, s in enumerate(mag_user_inputs):
        l = tk.Label(mag_tab, text=s, bg="gray60")
        ypos = (i+1) * lstep
        l.place(relx=xpos, rely=ypos, ancho="nw", relwidth=wid, relheight=0.05)
        _ = Hovertip(l, tools_tips[i], hover_delay=500)
        if s == "State":
            st = tk.StringVar()
            st.set("")
            d = ttk.Combobox(mag_tab, textvariable=st, values=list(RANKS.keys()))
            d.current(1)
            d.place(relx=(xpos+wid), rely=ypos, relwidth=wid, relheight=0.05)
            user_inputs[s] = d
        else:
            lim = input_lims[s]
            t = tk.Spinbox(mag_tab, from_=lim[0], to=lim[1],
                           increment=lim[2])
            t.place(relx=(xpos+wid), rely=ypos, relwidth=wid, relheight=0.05)
            user_inputs[s] = t

    # prediction outputs, similar setup to user inputs but on the other side
    outputs = ["EF 0 Confidence", "EF 1 Confidence", "EF 2 Confidence", "EF 3 Confidence", "EF 4 Confidence", "EF 5 Confidence"]
    tools_tips = ["Likeliness that tornado is EF0", "Likeliness that tornado is EF1", "Likeliness that tornado is EF2", "Likeliness that tornado is EF3", "Likeliness that tornado is EF4", "Likeliness that tornado is EF5"]
    output_labels = {}
    new_xpos = 0.5+xpos
    preds = tk.Label(mag_tab, text="Predictions", bg="LightGoldenrod1")
    preds.place(relx=(new_xpos+(wid/2)), rely=0.01, anchor="nw", relwidth=wid)
    for i, s in enumerate(outputs):
        l = tk.Label(mag_tab, text=s, bg="LightGoldenrod1")
        _ = Hovertip(l, tools_tips[i], hover_delay=500)
        ypos = (i+1) * lstep
        l.place(relx=new_xpos, rely=ypos, ancho="nw", relwidth=wid, relheight=0.05)
        o = tk.Label(mag_tab, text="", bg="white")
        o.place(relx=(new_xpos+wid), rely=ypos, relwidth=wid, relheight=0.05)
        output_labels[s] = o

    # buttons
    pred_button = tk.Button(mag_tab, text="Predict", 
                            command=lambda: _run_prediction_mag(user_inputs, output_labels),
                            bg="navy", fg="white", relief="raised")
    load_button = tk.Button(mag_tab, text="Load CSV", command=lambda: _load_csv_mag(pred_button),
                            bg="navy", fg="white", relief="raised")
    clear_button = tk.Button(mag_tab, text="Clear CSV Data", command=lambda: _clear_csv_data_mag(pred_button),
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
    _setup_mag_tab(mag_tab)

    window.mainloop()