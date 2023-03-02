# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 00:04:35 2023

@author: Kushum
"""

#Loading Library
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#Setting working directory
path ='D:\\OneDrive - Lamar University\\00Spring2023\\MachineLearning\\Project_1\\wd'
os.chdir(path)

#Step 1: Read the Compiled Data File
a = pd.read_csv('1.Finaldata.csv')
a.head()


#Step 2: Filtering the data
#Step 2.1: Filter by Latitude and longitude
a = a[(a['LON'] > -95.25) & (a['LON'] < -92.25) & (a['LAT'] > 27.25) & (a['LAT'] < 30.25)]

#Step 2.2: Filter the unavailable data
#drop the IMO, Draft column.
# count the number of missing values in the 'IMO' column
#num_missingIMO = a['IMO'].isna().sum()

# count the number of missing values in the 'Draft' column
#num_missingdraft = a['Draft'].isna().sum()

# count the number of missing values in the 'Cargo' column
#num_missingCargo = a['Cargo'].isna().sum()

# count the number of zero values in the 'Length' column
#no_length = a['Length'].isna().sum()
#no_width = a['Width'].isna().sum()

# As more than 55% of data are na in IMO and 62% data are na in draft
#drop the IMO column and Draft column
ad = a.drop(['IMO', 'Draft'], axis=1)

#Filter all the not available data from all columns
a1 = ad.dropna()

#Step 2.3: Filter the stationary vessels
a1['SOG'].plot()
plt.show()

#Count the SOG data less than 0.2
count = len(a1[a1['SOG'] < 0.5])
print(count)

a2 = a1[(a1['SOG'] > 0.5)]
a2['SOG'].plot()
plt.show()

#Step 2.4: Select the data to be used
#Data1 = a2.iloc[:,[0,1,2,3,4,5,6,7,8]]
Data1 = a2
 
#Step 2.5: Sorting by MMSI and then Date & Time
Data1 = Data1.sort_values(by=['MMSI','BaseDateTime'], ascending=True)
Data1.reset_index(drop=True)
Data1.head()

#Step 2.6: FIltering data for MMSI (having enough data for unique MMSI)



#Step 2.7: Print the filtered data to a csv file
#pd.DataFrame.to_csv(Data1,'2.FilterData.csv', index=False)


#Step 3: Compute correlations and plot correlation matrix (of the filtered data)
corr = Data1.corr(method='kendall')
#pd.DataFrame.to_csv(corr,'correl.csv') # Write to csv file
# source:https://seaborn.pydata.org/examples/many_pairwise_correlations.html

#Step 3.2: Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Step 3.3: Set up the matplotlib figure
f, Data1x = plt.subplots(figsize=(11, 9))

#Step 3.4: Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

#Step 3.5: Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


#Step 4: Step Pre-Processing for regression (Y variable)
#Step 4.1: Creating a new columns to calcuate diff in Latitude and longitude
Data1['LAT_shift'] = Data1['LAT'].shift(-1)
Data1['LON_shift'] = Data1['LON'].shift(-1)
Data1.head()

#Step 4.2: Cheking for unique MMSI and filtering complete data available for unique MMSI (Data2)(travel id)
Data1['MMSI_shift'] = Data1['MMSI'].shift(-1)
Data1.head()

Data1['MMSI_check'] = Data1['MMSI_shift'] - Data1['MMSI']
Data1.head()

Data2 = Data1[Data1['MMSI_check'] == 0]

#Step 4.3: Creating dataframe with latitude difference and longitude difference to find the direction of trajectory
Data2['LAT_diff'] = Data1['LAT_shift'] - Data1['LAT']
Data2['LON_diff'] = Data1['LON_shift'] - Data1['LON']

#Step 4.4: Defining Condition for the new category (direction of the trajectory for new location)
def LAT_indicator(LAT_diff):
    if LAT_diff > 0:
        return '1'
    else:
        return '0'
    
def LON_indicator(LON_diff):
    if LON_diff > 0:
        return '1'
    else:
        return '0'
    
#Step 4.5: Using apply and Lambda funtion to create new column for LAT and LON as the indicator (0,1)
Data2['LAT_indicator'] = Data2['LAT_diff'].apply(lambda LAT_diff: LAT_indicator(LAT_diff))
Data2['LON_indicator'] = Data2['LON_diff'].apply(lambda LON_diff: LON_indicator(LON_diff))


#Step 4.6: Printing the pre-prossed data to s csv file (to be used for developing and testing a linear regression model)
#pd.DataFrame.to_csv(Data2,'Data.csv', index=False)














