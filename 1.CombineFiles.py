# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 00:34:10 2023

@author: Kushum
"""

#Loading Library
import os
import pandas as pd
import glob


#Setting working directory
path ='D:\\OneDrive - Lamar University\\00Spring2023\\MachineLearning\\Project_1\\wd\\dataWD'
os.chdir(path)

#Get a list of all CSV files in the directory
csv_files = glob.glob(path + '/*.csv')

#Initialize an empty list to store the dataframes
dfs = []

# Loop over each CSV file, read it into a dataframe, and append it to the list
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all dataframes into a single dataframe
final_df = pd.concat(dfs, axis=0, ignore_index=True)

# Print the final dataframe
print(final_df)

#Setting working directory
path ='D:\\OneDrive - Lamar University\\00Spring2023\\MachineLearning\\Project_1\\wd'
os.chdir(path)
pd.DataFrame.to_csv(final_df,'1.Finaldata.csv', index=False)

#Check: Read the data File
#a = pd.read_csv('AIS_2022_01_15.csv')
#b = pd.read_csv('AIS_2020_01_15.csv')