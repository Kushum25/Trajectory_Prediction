# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 22:32:55 2023

@author: Kushum
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
import seaborn as sns

#Setting working directory
path ='D:\\OneDrive - Lamar University\\00Spring2023\\MachineLearning\\Project_1\\wd'
os.chdir(path)

#Step 1: Read the filtered data File (csv file)
a = pd.read_csv('Data.csv')
a.head()

#Filter for zero values of length
#no_length = a['Length'].isna().sum()  #check for na
num_zeros = (a['Length'] == 0).sum()

a1 = a[(a['Length'] > 0)]

# Step 2: Define the variables, Dataframe X and Y
X = a1.iloc[:,[2, 5, 11]]
Y = a1.iloc[:,15] 

#Step 3: Slipt the data into training and testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=9841)

#Step 4: FOR TRAINING DATASET Perform Regression using variables (4-7 steps)
#Step 4.2: Add a constant to the X data for the intercept term in the regression
X_train = sm.add_constant(X_train) 

#Step 4.3: Fit the model For Y latitude Y1
mod = sm.OLS(Y_train, X_train)     # Fit a multiple linear regression model
result = mod.fit()                  
print(result.summary())

#for error calculation
Y_pred = result.fittedvalues.copy()
Y_err = Y_train - Y_pred

#Step5: Compute Variance Inflation Factors (Calculate VIF for each independent variable)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

# Step 6: Plot for various tests
#Step 6.1: Run tests for LINE
plt.plot(Y_pred,Y_err,'ro')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.grid()

#Step 6.2: Obs-Predicted plot
plt.plot(Y_train,Y_pred,'bo')
plt.plot(Y_train, Y_train + 0, linestyle='solid')
plt.xlabel('Observed LATITUDE' )
plt.ylabel('Predicted LATITUDE' )
plt.grid()

#Step 6.3: Breusch Pagan Test for homoskedasticity
BP = sm.stats.diagnostic.het_breuschpagan(Y_err,X_train)

#Step 6.4: Tests of Normality
sm.qqplot(Y_err,line ='s')
plt.grid()

sm.stats.diagnostic.kstest_normal(Y_err, dist='norm')

#Step 6.5: Autocorrelation testing
sm.stats.diagnostic.acorr_breusch_godfrey(result) #need to provide regression model


# Step 7: Create a confusion Matrix
#Step 7.1:Normalize the data between 0 and 1 for Y_train and Y_predict
Y_train_norm=(Y_train-Y_train.min())/(Y_train.max()-Y_train.min())
Y_train1 = Y_train_norm 

Y_pred_norm=(Y_pred-Y_pred.min())/(Y_pred.max()-Y_pred.min())
Y_pred1 = Y_pred_norm 

#Step 7.2: Reclassify the normalized value to binary
#for Training dataset Y_train
threshold = 0.5
Y_REtrain = Y_train1.apply(lambda Y_train1: Y_train1 >= threshold).astype(int)

#for Predicted dataset Y_pred
threshold = 0.5
Y_REpred = Y_pred1.apply(lambda Y_pred1: Y_pred1 >= threshold).astype(int)

# Step 7.3: Create a confusion Matrix
cnf_matrix = metrics.confusion_matrix(Y_REtrain, Y_REpred)
cnf_matrix # y_test is going be rows (obs), y_pred (predicted) are cols

# Step 7.3.1: Evaluate usng accuracy, precision, recall
print("Accuracy:",metrics.accuracy_score(Y_REtrain, Y_REpred)) # overall accuracy
print("Precision:",metrics.precision_score(Y_REtrain, Y_REpred)) # predicting 0 (Sat)
print("Recall:",metrics.recall_score(Y_REtrain, Y_REpred)) # predicting 1 (unsat)

# Step 7.3.2:Plot confusion Matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# Step 7.3.3: Create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
ax.yaxis.set_label_position("left")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


#Step 8: For TESTING DATASET (step 8 -)
# STep 8.1: Fitting the trained model to testing dataset
# Add constant to the X-variable
X_test = sm.add_constant(X_test)
Y_test = Y_test

# Trained model Fit
Y_test_pred = result.predict(X_test)

#for error calculation
Y_test_err = Y_test - Y_test_pred

# Step 9: Plot for various tests
#Step 10.1: Run tests for LINE
plt.plot(Y_test_pred,Y_test_err,'ro')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.grid()

#Step 10.2: Obs-Predicted plot
plt.plot(Y_test,Y_test_pred,'bo')
plt.plot(Y_test, Y_test + 0, linestyle='solid')
plt.xlabel('Observed LATITUDE' )
plt.ylabel('Predicted LATITUDE' )
plt.grid()

#Step 10.3: Breusch Pagan Test for homoskedasticity
BP1 = sm.stats.diagnostic.het_breuschpagan(Y_test_err,X_test)

#Step 10.4: Tests of Normality
sm.qqplot(Y_test_err,line ='s')
plt.grid()

sm.stats.diagnostic.kstest_normal(Y_test_err, dist='norm')

#Step 10.5: Autocorrelation testing
sm.stats.diagnostic.acorr_breusch_godfrey(result) #need to provide regression model


# Step 11: Create a confusion Matrix
#Step 11.1:Normalize the data between 0 and 1 for Y_train and Y_predict
Y_test_norm=(Y_test-Y_test.min())/(Y_train.max()-Y_train.min())
Y_test1 = Y_test_norm 

Y_test_pred_norm=(Y_test_pred-Y_test_pred.min())/(Y_test_pred.max()-Y_test_pred.min())
Y_test_pred1 = Y_test_pred_norm 

#Step 11.2: Reclassify the normalized value to binary
#for Training dataset Y_train
threshold = 0.5
Y_REtest = Y_test1.apply(lambda Y_test1: Y_test1 >= threshold).astype(int)

#for Predicted dataset Y_pred
threshold = 0.5
Y_REpred2 = Y_test_pred1.apply(lambda Y_test_pred1: Y_test_pred1 >= threshold).astype(int)

# Step 11.3: Create a confusion Matrix
cnf_matrix = metrics.confusion_matrix(Y_REtest, Y_REpred2)
cnf_matrix # y_test is going be rows (obs), y_pred (predicted) are cols

# Step 11.4: Evaluate usng accuracy, precision, recall
print("Accuracy:",metrics.accuracy_score(Y_REtest, Y_REpred2)) # overall accuracy
print("Precision:",metrics.precision_score(Y_REtest, Y_REpred2)) # predicting 0 (Sat)
print("Recall:",metrics.recall_score(Y_REtest, Y_REpred2)) # predicting 1 (unsat)


# Step 11.5: Plot confusion Matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
ax.yaxis.set_label_position("left")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')



