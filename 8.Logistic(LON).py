# -*- coding: utf-8 -*-
"""
Created on Feb 23 16:26:12 2023

@author: Rakshya
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns

#Setting working directory
path ='C:/Users/14098/OneDrive - Lamar University/Desktop/Machine Learning/Project 1/AIS_2022'
os.chdir(path)

#Step 1: Read the filtered data File (csv file)
a = pd.read_csv('Data.csv')
a.head()

# Step 2: Define the variables, Dataframe X and Y
X = a.iloc[:,[2,4,5,6]] 
Y = a.iloc[:,22] 

#Normalize the between 0 and 1 for all X variables (Reclassify for Loglogis)
X_norm=(X-X.min())/(X.max()-X.min())
X = X_norm #FOr log logistic

#Step 3: Slipt the data into training and testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=234)

#Step 4: For TRAINING DATASET, Perform Regression using same variables as for Linear Model
#Step 4.1: Logisitic Regression
Y_train = Y_train
Y_test = Y_test

#Step 4.2: Fitting the model
logreg = LogisticRegression(C=10**9)
logreg.fit(X_train, Y_train)

# training data
y_pred=logreg.predict(X_train) # fit testing data
yprob = logreg.predict_proba(X_train) #output probabilities
zz = pd.DataFrame(yprob)
zzz = pd.DataFrame(yprob)
#zz.to_csv('probs.csv')

#Step 4.3: Get the parameters
logreg.get_params()
logreg.coef_
logreg.intercept_

#Step 4.4: Create a confusion Matrix
cnf_matrix = metrics.confusion_matrix(Y_train, y_pred)
cnf_matrix # y_test is going be rows (obs), y_pred (predicted) are cols

# Step 4.5: Evaluate usng accuracy, precision, recall
print("Accuracy:",metrics.accuracy_score(Y_train, y_pred)) # overall accuracy
print("Precision:",metrics.precision_score(Y_train, y_pred)) # predicting 0 (Sat)
print("Recall:",metrics.recall_score(Y_train, y_pred)) # predicting 1 (unsat)

#Step 4.6: ROC Curve
y_pred_proba = logreg.predict_proba(X_train)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_train,  y_pred_proba)
auc = metrics.roc_auc_score(Y_train, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(round(auc,4)))
plt.legend(loc=4)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.grid()
plt.show()

# Step 4.5.1: Plot confusion Matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
ax.yaxis.set_label_position('left')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# Step 5: Fit the model to the testing data
y_pred=logreg.predict(X_test) # fit testing data
yprob = logreg.predict_proba(X_test) #output probabilities
zz = pd.DataFrame(yprob)
zzz = pd.DataFrame(yprob)
#zz.to_csv('probs.csv')

# Get the parameters
logreg.get_params()
logreg.coef_
logreg.intercept_

# Create a confusion Matrix
cnf_matrix = metrics.confusion_matrix(Y_test, y_pred)
cnf_matrix # y_test is going be rows (obs), y_pred (predicted) are cols

# Evaluate usng accuracy, precision, recall
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred)) # overall accuracy
print("Precision:",metrics.precision_score(Y_test, y_pred)) # predicting 0 (Sat)
print("Recall:",metrics.recall_score(Y_test, y_pred)) # predicting 1 (unsat)

# ROC Curve
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(round(auc,4)))
plt.legend(loc=4)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.grid()
plt.show()

# Plot confusion Matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
ax.yaxis.set_label_position('left')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')