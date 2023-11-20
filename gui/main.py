import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pickle
import pandas as pd

CSV_PATH = "data\\tornado_wind_data.csv"
WIND_CSV_PATH = "data\\wind.csv"
DATA_LOADED = False

# prediction button response
def _run_prediction():
    _ = messagebox.showinfo("Prediction", "Predicting...")

# load csv button response
def _load_csv():
    _ = messagebox.showinfo("Load CSV", "Loading...")

# clear csv button response
def _clear_csv_data():
    DATA_LOADED = False
    _ = messagebox.showinfo("Clear CSV", "Cleared!")

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
    # get wind data and match to parameters
    wind = pd.read_csv(WIND_CSV_PATH)
    data = wind[wind["st"] == state & wind["mo"] == month & wind["dy"] == day]
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
        data = wind[wind["st"] == state]
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
    cas_user_inputs = ["Month","Day","Time","State","Magnitude","Track Legnth (mi)","Storm Width (yd)",
                   "Latitude", "Longitude"]
    
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
        t = tk.Text(cas_tab)
        t.place(relx=(xpos+wid), rely=ypos, relwidth=wid, relheight=0.05)
        user_inputs[s] = t

    # buttons
    load_button = tk.Button(cas_tab, text="Load CSV", command=_load_csv,
                            bg="navy", fg="white", relief="raised")
    pred_button = tk.Button(cas_tab, text="Predict", command=_run_prediction,
                            bg="navy", fg="white", relief="raised")
    clear_button = tk.Button(cas_tab, text="Clear CSV Data", command=_clear_csv_data,
                            bg="tomato", relief="raised")
    pred_button.place(relx=0.5, rely=0.9, relwidth=wid)
    load_button.place(relx=0.5, rely=0.8, relwidth=wid)
    clear_button.place(relx=0.5, rely=0.7, relwidth=wid)

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

if __name__ == "__main__":
    # DATA
    with open("data\\state_ranks.pkl", "rb") as f:
        ranks = pickle.load(f)

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