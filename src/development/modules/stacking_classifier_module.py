"""
This script combines the outputs of multiple models to create a final prediction. This is done by using a meta learner to combine the outputs of the base learners. This si known as stacking.

The model hyperparamters were trained in the previous script using the Optuna library. We will now be applying the hyperparameters learned in the previous script, and applying them to train models on the entire training dataset. 

The models that will be used are:
    - Random Forest
    - XGBoost
    - k-NN Classifier
    - SVM (Support Vector Machine)
    - Naive Bayes Network (PGMPY Implemntation)
    - Neural Network (Keras Implemntation)
    - CatBoost (Look into this)
    
I will be using the scikit-learn library to implement the stacking classifier.
(https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)

Finally different meta learners will be used to combine the outputs of the base learners. The meta learners that will be used are:
    - Logistic Regression
    - Random Forest
    - XGBoost

The meta learners will be trained on the outputs of the base learners. The outputs of the base learners will be the probability outputs of the models. This is because the probability outputs provide more context on the decisions being made.
(https://towardsdatascience.com/a-deep-dive-into-stacking-ensemble-machine-learning-part-i-10476b2ade3#:~:text=In%20a%20binary%20classification%20for,the%20stacking%20model%20improves%20significantly.)

Maybe look at using a non-timseries cross validation. I know this isn't somethign I should ideally do but it allows me to use entire dataset and may not impact the model too much.
Some people seem to be saying that the meta model must be trained on a seperate dataset than the datastes used to train the base models. This is to prevent poor generalisation because of target leak.
https://stats.stackexchange.com/questions/239445/how-to-properly-do-stacking-meta-ensembling-with-cross-validation?rq=1

TODO: Maybe look at putting game_id in a type of lookup. JUst so I'm completely dure that I'm joinging the correct data together.In the stakced probabilities part of the code
"""

#Imports

#Directory management
import os
from pathlib import Path  
import pyarrow.feather as feather   # Package to store dataframes in a binary format
import joblib
import copy as cp
import numpy as np

#Model preparation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

#Base models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from pgmpy.models import BayesianNetwork
from tensorflow import keras

#LSTM
#Stacking models
from sklearn.ensemble import StackingClassifier

#Hyperparameter tuning
import optuna   #You can get a progress bar for optuna
import optuna_dashboard

#Model evaluation
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_score

os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")

intermediate = Path.cwd() / "data" / "intermediate"

####################################
#Prepping Input Data
####################################

#Cross validation
tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)

#Look up walk forward validation (https://sarit-maitra.medium.com/take-time-series-a-level-up-with-walk-forward-validation-217c33114f68)
#https://www.linkedin.com/pulse/walk-forward-validation-yeshwanth-n/
#https://github.com/scikit-learn/scikit-learn/issues/8043
#https://stats.stackexchange.com/questions/483888/cross-validation-in-stackingclassifier-scikit-learn

#Loading the data
#Loading in feature selected columns
fs_columns = joblib.load(f"{intermediate}/fs_columns.save")
#fs_columns = joblib.load(f"{intermediate}/fs_columns.pkl")

#Reading in the data that will be used in the model
input_data = pd.read_feather(f"{intermediate}/input_data.feather")

#Sorting index to make sure that the data is in chronological order
input_data = input_data.sort_index()

#Reducing the data to only the feature selected columns and the result column (This is the target variable)
input_data_fs = input_data[fs_columns + ["result"]]

#I will be using cross validation but I need to make sure that I hold out some test data for the final testing of the model. This is what I'm splitting out at this stage
#I'm holding back 10% of the data for final testing of the model
X_train, X_test, y_train, y_test = train_test_split(input_data_fs.drop(columns=["result"], axis=1), input_data_fs["result"], 
                                                    test_size=0.10, shuffle=False) #Setting shuffle to false, again to prevent data leakage

#Scaling the data. This scaler will also be applied to the test data
scaler = StandardScaler() #Creating the scaler object

joblib.dump(scaler, f"{intermediate}/scaler.pkl") #Saving the scaler object so it can be used on the test data

scaler.fit(X_train) #Fitting the scaler object to the training data
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index = X_train.index)#scaling the training data

####################################
#Stacking Models
####################################
#This method for stacking ensemble has been tacken from https://towardsdatascience.com/a-deep-dive-into-stacking-ensemble-machine-learning-part-ii-69bfc0d6e53d


#Loading the study for each of the models
rf_study = joblib.load(f"data/intermediate/model_studies/rf_study.pkl")
gb_study = joblib.load(f"data/intermediate/model_studies/gb_study.pkl")
svc_study = joblib.load(f"data/intermediate/model_studies/svc_study.pkl")
knn_study = joblib.load(f"data/intermediate/model_studies/knn_study.pkl")

#Overwrtiting the best parameters for the SVC model to include the probability parameter

#Level 0 has been implemented as a dictionary of base classification models (This will be added to as more models are added to the ensemble)
level_0_classifiers = dict()
level_0_classifiers["rf"] = RandomForestClassifier(**rf_study.best_params, random_state=41)
level_0_classifiers["gb"] = GradientBoostingClassifier(**gb_study.best_params, random_state=41)
#level_0_classifiers["svc"] = SVC(**svc_study.best_params, random_state=41) #Will put back in once I've retrained the model with the probability parameter
level_0_classifiers["knn"] = KNeighborsClassifier(**knn_study.best_params)

#Using a random forest classifier as the level 1 classifier
level_1_classifier = RandomForestClassifier(random_state=41)

#An issue with using time series data is that we will lose extra data as the first segment is never predicted and the last segment is never used for training.
#This should help for cross validation stacking (https://datascience.stackexchange.com/questions/41378/how-to-apply-stacking-cross-validation-for-time-series-data)
#I can probably also code this myselfusing somethign liek this

#Instantiating a list of dataframes that will be stuck togther (top to bottom) at the end
stacked_probabilities_list = [] #This will be a list of dataframes that will be concatenated together at the end

for trial, (train_index, test_index) in enumerate(tscv.split(X_train)):
    
    print(f"Trial {trial}")
    
    X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]
    
    stacked_probabilities_df = pd.DataFrame()

    for name, classifier in level_0_classifiers.items():
        classifier_ = cp.deepcopy(classifier)
        classifier_.fit(X_train_cv, y_train_cv)
        
        y_predict_proba = classifier_.predict_proba(X_test_cv)
                
        # Get the number of classes from the shape of y_predict_proba
        class_names = classifier_.classes_
        
        # Add a new column for each class
        for i, class_name in enumerate(class_names):
            stacked_probabilities_df = pd.concat([stacked_probabilities_df, pd.DataFrame(y_predict_proba[:, i], columns=[f"{name}_{class_name}_prediction"])], axis=1)
            
    #Adding a column to show which rows have been used
    stacked_probabilities_df = pd.concat([stacked_probabilities_df, pd.DataFrame(test_index, columns = ["index_number"])], axis=1)
    stacked_probabilities_list.append(stacked_probabilities_df)
            
            

#Concatenating the dataframes together (top to bottom)
stacked_probabilities_df = pd.concat(stacked_probabilities_list, axis=0)

#Setting the index to the index number

#Creating new column that can be used to merge the predictions back onto the data
X_train_stacked = cp.deepcopy(X_train)

#Create a new column indicating the row number of that row called index_number. This will be used to merge the predictions back onto the data
X_train_stacked = X_train_stacked.assign(index_number=range(len(X_train_stacked)))

# Save the date index in a separate column
X_train_stacked['date'] = X_train_stacked.index

# Perform the merge
X_train_stacked = pd.merge(X_train_stacked, stacked_probabilities_df, on="index_number", how="inner")

# Set the date column as the index again
X_train_stacked.set_index('date', inplace=True)

#Now applying the level 1 classifier to the stacked probabilities dataset. dropping the index_number column as this is no longer needed for the modelling

#reducing y_train down to the same rows as X_train_stacked
y_train_stacked = pd.DataFrame(y_train).assign(index_number=range(len(y_train)))

#reducing y_train down to the same rows as X_train_stacked. 
y_train_stacked = y_train_stacked[y_train_stacked["index_number"].isin(X_train_stacked["index_number"])]


#Fitting the level 1 classifier (meta-Learner) to the base input features + the stacked probabilities dataset (Should I be using some type of cross validation here?)
level_1_classifier.fit(X_train_stacked.drop(labels=["index_number"], axis=1), y_train_stacked.drop(labels=["index_number"], axis=1))

#We now have a trained stacked meta model. This can now be applied to test data to see how well it compares to the base models
 
#First we need to create the stacked probabilities for the test data
stacked_probabilities_df_test = pd.DataFrame()

#Scaling the test data. The same scaler is being used as that was used on the training data
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index = X_test.index)#scaling the training data

for name, classifier in level_0_classifiers.items():
    classifier_ = cp.deepcopy(classifier)
    classifier_.fit(X_train, y_train)

    y_predict_proba = classifier_.predict_proba(X_test_scaled)
            
    # Get the number of classes from the shape of y_predict_proba
    class_names = classifier_.classes_
    
    # Add a new column for each class
    for i, class_name in enumerate(class_names):
        stacked_probabilities_df_test = pd.concat([stacked_probabilities_df_test, pd.DataFrame(y_predict_proba[:, i], columns=[f"{name}_{class_name}_prediction"])], axis=1)
        
        
#Merging the stacked probabilities with the test data
#Firstly adding a column with the row numbers that I will be using to merge
stacked_probabilities_df_test = stacked_probabilities_df_test.assign(index_number = range(len(stacked_probabilities_df_test)))
X_test_scaled = X_test_scaled.assign(index_number = range(len(X_test_scaled)))

#Joining the data together
X_test_stacked = pd.merge(X_test_scaled, stacked_probabilities_df_test, on="index_number", how="inner")

    
#Checking that we have the same columns in the test data as we do in the training data
#checking that all the columns are the same
if sum(X_train_stacked.columns != X_test_stacked.columns) != 0:
    print("The test and training data do not have the same columns")
    
#Now applying the level 1 classifier to the stacked probabilities dataset. dropping the index_number column as this is no longer needed for the modelling   

#Predicting the outcome of the test data
y_pred_stacked = level_1_classifier.predict(X_test_stacked.drop(labels=["index_number"], axis=1)) 

#converting the predictions to a dataframe
y_pred_stacked_df = pd.DataFrame(y_pred_stacked, columns=["prediction"], index=X_test_stacked.index)
y_pred_stacked_df.value_counts(normalize=True)
y_test.value_counts(normalize=True)
#classification report
print(classification_report(y_test, y_pred_stacked))

#Getting the feature importances of the stacked model and the names of the features
stacked_feature_importances = level_1_classifier.feature_importances_



stacking_classifier = StackingClassifier(estimators=list(level_0_classifiers.items()), final_estimator=level_1_classifier, cv=tscv, passthrough=True, stack_method="predict_proba")

#Creating column names for all the level 0 base classifiers
level_0_columns = [f"{name}_prediction" for name in level_0_classifiers.keys()]
pd.DataFrame(stacking_classifier.fit_transform(X_train, y_train), columns=level_0_columns + list(X_train.columns))


test = pd.DataFrame(stacking_classifier.fit_transform(X_train, y_train))
####################################
#Training Base Models
####################################

#Generating prediction from the randomforest model
rf_model = RandomForestClassifier(**rf_study.best_params, random_state=41)

#Predicting the outcome of the test data
rf_model.fit(X_train, y_train)
rf_predictions_prob = pd.DataFrame(rf_model.predict_proba(X_test))

#Creating new column that gives the class prediction
rf_predictions_prob["prediction"] = rf_model.predict(X_test)

#Getting the best scores from the study
rf_study.best_value

y_train.value_counts(normalize=True)

rf_predictions_prob["prediction"].value_counts(normalize=True)

#adding y_test predictions to the dataframe
rf_predictions_prob = pd.concat([rf_predictions_prob, pd.DataFrame(y_test).reset_index(drop=True)], axis=1)

#When class are equal then flagging 1, else 0
#when prediction is equal to the result then flagging 1, else 0
rf_predictions_prob["correct_prediction"] = rf_predictions_prob["prediction"] == rf_predictions_prob["result"]  

rf_predictions_prob['correct_prediction'] = rf_predictions_prob.apply(lambda row: 1 if row["prediction"] == row["result"] else 0, axis=1)

rf_predictions_prob['correct_prediction'].mean()


rf_predictions_prob["correct_prediction"] = rf_predictions_prob["prediction"] == rf_predictions_prob["result"]

rf_predictions_prob["prediction"].value_counts()

y_test.value_counts(normalize=True)   

#Concatenating the prediction probabilities with the actual result
rf_predictions_prob = pd.concat([rf_predictions_prob, pd.DataFrame(y_test)], axis=1)

#Creating a classification report
print(classification_report(y_test, rf_predictions_prob["prediction"]))


rf_predictions_prob = pd.merge(rf_predictions_prob, pd.DataFrame(y_test), left_index=True, right_index=True)


rf_predictions_prob.columns = rf_model.classes_


#Each of the individual models will now be trained on the entire dataset using the best parameters found above (Maybe done using stacking classifier)
# rf_best_model.fit(X_train, y_train)
# gb_best_model.fit(X_train, y_train)
# svc_best_model.fit(X_train, y_train)
# knn_best_model.fit(X_train, y_train)
