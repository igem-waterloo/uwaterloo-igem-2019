import numpy as np 
import matplotlib.pyplot as pl 
import pandas as pd 
import scipy.optimize as sc 

#Reads the csv file into an array
data = pd.read_csv("ecdata_labelled_c3.csv")
timeSeries_raw = np.array(data['Time'])
timeSeries = np.array([])

#Converts the time column into hours
for t in timeSeries_raw:
    parts = t.split(":")
    seconds = int(parts[0])*(60*60) + int(parts[1])*60 + int(parts[2])
    timeSeries = np.append(timeSeries,seconds)
timeSeries=timeSeries[40:114]/3600
#-----------------------------------

#Converts remaining columns into a numpy array
dataPoints=[]
for s in data.columns:
    if s != 'Time':
        dataPoints.append(np.array(data[s])[40:114]) #Format: A:B where A is the beginning timepoint and B is the final timepoint
dataPoints=np.array(dataPoints)
#--------------------------------------------

#Function to fit
def exponential(x,a,b,c):
    return a*np.exp(b*x)+c
#---------------

fits=[]
rsquar=[]
dtime = []

for i in range(0,len(dataPoints)):
    #Fit the curve
    popt, pcov = sc.curve_fit(exponential,timeSeries,dataPoints[i])
    residuals = dataPoints[i]-exponential(timeSeries,popt[0],popt[1],popt[2])

    #Compute R^2,
    SS_res=np.sum(residuals**2)
    SS_tot=np.sum((dataPoints[i]-np.mean(dataPoints[i]))**2)
    rsquared=1-SS_res/SS_tot

    #Save data to array
    rsquar.append(np.array(rsquar))
    fits.append(np.array(popt))
    dtime.append(np.log(2)/fits[i][1])

Dval = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]

#Converts result into a table of D-Value versus doubling time (in hours)
table=[]
for i in range(0,len(Dval)):
    table.append(np.array([Dval[i],dtime[i]]))
#----------------------------------------------------------------------

print(table)
#Save array to a txt file
np.savetxt("fits_c3.txt",table,delimiter=", ")

