""" Implements a class that performs k nearest neighbor classificatoin.
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random

class KNearestClassifier() : 
    # store data set that will be used to find neighbors
    def __init__(self, df): 
        # store month, Length of path, Width of path,  time on ground, starting latitude, starting longitude, maximum wind gust speed, EF rating of tornadoes
        self.params = df.filter(['mo', 'len', 'wid', 'seconds', 'slat', 'slon', 'max_gust', 'mag'], axis =1)

        # replace NaN values of gusts with mode for each EF rating #TODO try mean
        # split dataframe by ef value
        EF1Data = self.params.loc[df["mag"] == 0 ]
        EF1Data = [gust for gust in EF1Data.max_gust if str(gust) != 'nan' and gust != 0]
        EF2Data = self.params.loc[df["mag"] == 1 ]
        EF2Data = [gust for gust in EF2Data.max_gust if str(gust) != 'nan' and gust != 0]
        EF3Data = self.params.loc[df["mag"] == 2 ]
        EF3Data = [gust for gust in EF3Data.max_gust if str(gust) != 'nan' and gust != 0]
        EF4Data = self.params.loc[df["mag"] == 3 ]
        EF4Data = [gust for gust in EF4Data.max_gust if str(gust) != 'nan' and gust != 0]
        EF5Data = self.params.loc[df["mag"] == 4 ]
        EF5Data = [gust for gust in EF5Data.max_gust if str(gust) != 'nan' and gust != 0]

        # replace in NaN main data frame
        for row in self.params.itertuples():
            if row[8] == 0 and np.isnan(row[7]): self.params.at[row[0], 'max_gust'] = sum(EF1Data) / len(EF1Data)
            if row[8] == 1 and np.isnan(row[7]): self.params.at[row[0], 'max_gust'] = sum(EF2Data) / len(EF2Data)
            if row[8] == 2 and np.isnan(row[7]): self.params.at[row[0], 'max_gust'] = sum(EF3Data) / len(EF3Data)
            if row[8] == 3 and np.isnan(row[7]): self.params.at[row[0], 'max_gust'] = sum(EF4Data) / len(EF4Data)
            if row[8] == 4 and np.isnan(row[7]): self.params.at[row[0], 'max_gust'] = sum(EF5Data) / len(EF5Data)

        # store maximum values of paramters
        self.monthMax = max(abs(self.params['mo'])) 
        self.lenMax = max(abs(self.params['len'])) 
        self.widMax = max(abs(self.params['wid'])) 
        self.timeMax = max(abs(self.params['seconds'])) 
        self.slatMax = max(abs(self.params['slat'])) 
        self.slonMax = max(abs(self.params['slon'])) 
        self.gustMax = max(abs(self.params['max_gust']))  

    def classify(self, monthIn, lengthIn, widthIn, time, latIn, lonIn, gustIn):
        K=11 
        neighborDists = [100] * K
        neighborClass = [0] * K # 0=ef1,1=ef2,2=ef3,3=ef4,4=ef5
        #interate through stored params
        for row in self.params.itertuples():
            #print(row[0]) #print index TODO REMOVE
            #find distance to paramters and normalize 
            distance = abs(monthIn-row[1])/self.monthMax
            distance += abs(lengthIn-row[2])/self.lenMax
            distance += abs(widthIn-row[3])/self.widMax 
            distance += abs(time-row[4])/self.timeMax 
            distance += abs(latIn-row[5])/self.slatMax
            distance += abs(lonIn-row[6])/self.slonMax 
            distance += abs(gustIn-row[7])/self.gustMax
            #replace highest distance and its class if closer neighbor found
            highestIndex = np.argmax(neighborDists)
            if distance < neighborDists[highestIndex]:
                neighborDists[highestIndex] = distance
                neighborClass[highestIndex] = row[8] #magnitude

        return neighborClass
    
    
if __name__ == "__main__":
    df = pd.read_csv("data\\tornado_wind_data.csv")
    #test = KNearestClassifier(df)
    #test.classify(12, 2.13, 360, 59700, 43.5884, -110.5228, 87) #final row

    # Algorithm Evaluation
    # Split the data set 80/20
    training_set = df.sample(frac = 0.99)
    eval_set = df.drop(training_set.index)

    # Create classifier
    classifier = KNearestClassifier(training_set)

    #iterate eval_set in the classifier. count number of accurate classifications
    numberCorrect = [0,0,0,0,0] #store number correct per EF value
    count = 0
    for i, row in eval_set.iterrows():
        count += 1
        print (str(count) + "/" + str(len(eval_set))) #print progress
        magnitudes = classifier.classify(row.mo, row.len, row.wid, row.seconds, row.slat, row.slon, row.max_gust)
        #find most common magnitude
        EFcounts = [0,0,0,0,0] #store occurances of each ef val
        for mag in magnitudes: 
            if mag > 4: continue
            EFcounts[mag] += 1
        mag_pred = np.argmax(EFcounts) # mode of magnitudes. if tie use lower EF
        if (mag_pred == row.mag): 
            numberCorrect[mag_pred] += 1
            print("correct " + str(mag_pred))

    print("EF1 accuracy")
    print(numberCorrect[0]/len(eval_set.loc[df["mag"] == 0 ]))
    print("EF2 accuracy")    
    print(numberCorrect[1]/len(eval_set.loc[df["mag"] == 1 ]))
    print("EF3 accuracy")
    print(numberCorrect[2]/len(eval_set.loc[df["mag"] == 2 ]))
    print("EF4 accuracy")
    print(numberCorrect[3]/len(eval_set.loc[df["mag"] == 3 ]))
    print("EF5 accuracy")
    print(numberCorrect[4]/len(eval_set.loc[df["mag"] == 4 ]))
    print("Overall accuracy")
    print(sum(numberCorrect)/len(eval_set))