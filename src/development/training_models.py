"""
This script takes the pre-processed input data and trains multiple classfication models on said data.

These models will then be used in a stacked ensemble model to predict the final output.
This model will then be stored so that it can be used later on. I also might want to retrain the model as I get new data as more games are played.

https://developer.ibm.com/articles/stack-machine-learning-models-get-better-results/ (This is good for model stacking)
http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/ (This is about cross validation stacking for classifiers)
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html


Apparently (according to the using bookmaker odds to predict football outcomes paper) the best model for draws are the ensemble-selection classifier

"""
import os
from pathlib import Path  
import pyarrow.feather as feather   # Package to store dataframes in a binary format

import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
import pickle
import optuna   
from tensorflow import keras

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")


#Project directory locations
raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"   



####################################
#Reading in the data
####################################

all_football_data = pd.read_feather(f"{intermediate}/all_football_data.feather")

#setting the index to the date
all_football_data = all_football_data.set_index("date")

#Splitting data into test and train sets via a date split


X = all_football_data.drop(["result"], axis=1)
y = all_football_data["result"]

tscv = TimeSeriesSplit(n_splits=5)
rf = RandomForestClassifier(n_estimators=100, min_samples_split=10,random_state=42)

scores = cross_val_score(rf, X, y, cv=tscv, scoring='accuracy')

#Doing a manual time series split
train = all_football_data[all_football_data.index <= "2022-11-01"]
test = all_football_data[all_football_data.index > "2022-11-01"]

rf.fit(train.drop(["result"], axis=1), train["result"])

preds = rf.predict(test.drop(["result"], axis=1))


combined = pd.DataFrame({"preds": preds, "actual": test["result"]})


#Finidng occasions when the home and away prediction are the same
combined = pd.merge(combined, test[["home_team", "away_team"]], left_index=True, right_index=True)




accuracy_score(test["result"], preds)

#Confusion matrix of the results
confusion_matrix(test["result"], preds)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(test["result"], preds), display_labels=rf.classes_)
cm_display.plot()
