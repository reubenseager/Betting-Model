"""
    The purpose of the script is to look at which features to select for the model. This should not need to be ru each time, and so is not included in the primary processing.
"""

# Imports
# Math and data manipulation imports
import os
from pathlib import Path

import pandas as pd  # Package for data manipulation and analysis
import numpy as np # Package for scientific computing
import pyarrow.feather as feather   # Package to store dataframes in a binary format

#Machine Learning Imports
#Train test split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.linear_model import Lasso

#Saling
from sklearn.preprocessing import StandardScaler
import joblib


#Feature Selection
from sklearn.feature_selection import RFE #Not using the CV version as I am using time series data. Not sure if this is right though.
from sklearn.feature_selection import RFECV


os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")


#Project directory locations
raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"


####################################
#Feature Selection
###################################
#TODO = Look at using Genetic algorithms for feature selection (https://towardsdatascience.com/feature-selection-with-genetic-algorithms-7dd7e02dd237#:~:text=Genetic%20algorithms%20use%20an%20approach,model%20for%20the%20target%20task.)

#Reading in the data that will be used in the model

input_data = feather.read_feather(f"{intermediate}/all_football_data.feather")

#Set the date as the index. Also reorder the index so the most recent data is at the top. This is so that when we split the data into test and train sets, we don't use future data to predict past data.
input_data = input_data.sort_values(by="date", ascending=True)
input_data.set_index("date", inplace=True)

#Saving prescaled input data using feather
feather.write_feather(input_data, f"{intermediate}/input_data.feather")

#Splitting the data into test and train datasets. This needs to be done before any feature selection is done to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(input_data.drop(columns=["result"], axis=1), input_data["result"], test_size=0.15, shuffle=False) #Setting shuffle to false, again to prevent data leakage

#scaling the numerical features of the input data. Need to make sure that I apply the same scaler to the test data s well to avoid data leakage. Also this is what the models will be built off
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

#writing the scaler to a file so that it can be used later on in the process
joblib.dump(scaler, f"{intermediate}/scaler.save")

#counting occurences of each class in the target variable. This helps to determine which metric to use for the feature selection
y_train.value_counts()

#Now that the data has been scaled, I am going to apply some feature selection techqniques to try find the best subset of features to use in the model.
tscv = TimeSeriesSplit(n_splits=3) #Not entirely sure hwo big this should be (Maybe look up pros and cons of different split numbers)

#https://towardsdatascience.com/superior-feature-selection-by-combining-multiple-models-607002c1a324

#GradientBoosting Feature Selection
rfe = RFECV(estimator=GradientBoostingClassifier(random_state=41),
            step=1,
            min_features_to_select=15,
            cv=tscv,
            scoring="f1_macro") #Need to use an averaging metric for the scoring as the model is a multiclass model. Using f1_macro as it gives equal weight to each class.
            
_ = rfe.fit(X_train_scaled, y_train)

#Extracting a boolean mask, with True values being given for the features that should be kept
gb_mask = rfe.support_

#printing the features that should be kept
print(X_train_scaled.columns[gb_mask])

#RandomForest Feature Selection
rfe = RFECV(estimator=RandomForestClassifier(random_state=41),
            step=1,
            min_features_to_select=15,
            cv=tscv,
            scoring="f1_macro")

_ = rfe.fit(X_train_scaled, y_train)

#Extracting a boolean mask, with True values being given for the features that should be kept
rf_mask = rfe.support_

#printing the features that should be kept
print(X_train_scaled.columns[rf_mask])
print(X_train_scaled.columns[~rf_mask])


#Getting the score from the model (Not sure if will work)
params = rfe.get_params()

#Getting the score from the model (Not sure if will work)
print(rfe.score(X_train_scaled, y_train))


#SVC Feature Selection
rfe = RFECV(estimator=SVC(kernel="linear", random_state=41),
            step=1,
            min_features_to_select=15,
            cv=tscv,
            scoring="f1_macro")

_ = rfe.fit(X_train_scaled, y_train)

#Extracting a boolean mask, with True values being given for the features that should be kept
svc_mask = rfe.support_

#printing the features that should be kept
print(X_train_scaled.columns[svc_mask])
print(X_train_scaled.columns[~svc_mask])


#Combining the masks of the three models to vote on which features should be kept
feature_mask = np.sum([gb_mask, rf_mask], axis=0)

#Only selecting features that appeared in 2 of the three models
feature_mask = feature_mask >= 1

#Printing the features that should be kept
print(X_train_scaled.columns[feature_mask])

#printing the features that should be dropped
print(X_train_scaled.columns[~feature_mask])

#Storing the names of the features that should be kept in a list
fs_columns = X_train_scaled.columns[feature_mask].tolist()

joblib.dump(fs_columns, f"{intermediate}/fs_columns.pkl")
