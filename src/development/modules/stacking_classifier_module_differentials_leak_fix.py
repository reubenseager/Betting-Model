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
 I am taking a lot of the logic from this article: https://towardsdatascience.com/a-deep-dive-into-stacking-ensemble-machine-learning-part-i-eb317fcc3313

Finally different meta learners will be used to combine the outputs of the base learners. The meta learners that will be used are:
    - Logistic Regression
    - Random Forest
    - XGBoost

The meta learners will be trained on the outputs of the base learners. The outputs of the base learners will be the probability outputs of the models. This is because the probability outputs provide more context on the decisions being made.
(https://towardsdatascience.com/a-deep-dive-into-stacking-ensemble-machine-learning-part-i-10476b2ade3#:~:text=In%20a%20binary%20classification%20for,the%20stacking%20model%20improves%20significantly.)

Maybe look at using a non-timseries cross validation. I know this isn't somethign I should ideally do but it allows me to use entire dataset and may not impact the model too much.
Some people seem to be saying that the meta model must be trained on a seperate dataset than the datastes used to train the base models. This is to prevent poor generalisation because of target leak.
https://stats.stackexchange.com/questions/239445/how-to-properly-do-stacking-meta-ensembling-with-cross-validation?rq=1

#Look up walk forward validation (https://sarit-maitra.medium.com/take-time-series-a-level-up-with-walk-forward-validation-217c33114f68)
#https://www.linkedin.com/pulse/walk-forward-validation-yeshwanth-n/
#https://github.com/scikit-learn/scikit-learn/issues/8043
#https://stats.stackexchange.com/questions/483888/cross-validation-in-stackingclassifier-scikit-learn

TODO: Maybe look at putting game_id in a type of lookup. JUst so I'm completely dure that I'm joinging the correct data together.In the stakced probabilities part of the code
TODO: Maybe look at adjusting the thresholds for the models to see fi this changes how well the model performs.
TODO: Need to look at calibrating the classification of the stacked model. It's not predicting draws very well at the moment. they are more difficult to predict and are also rarer.

TODO: Implement hyperparameter tuning for the final meta learner. This means I need ot split my data into a validationa and true unseen test set.
TODO: Model isn't very good now. Didn't get any draw predictions correct. Need to look at calibrating the model and trying different meta learners. Also look at using the actual xgboost package instead of the scikit-learn wrapper
TODO: Maybe look at adding back the rolling numbers from the previous season like for gf and stuff.

#This paper seems quite good and compares lots of different models very well.
https://arno.uvt.nl/show.cgi?fid=147179
#I see a lot of papers use ranked probability score as a metric. Maybe look at using this instead.(https://github.com/GnyEser/RankedProbabilityScore/blob/master/rps.py)

"""

#Imports

#Directory and file management
import os
from pathlib import Path  
import pyarrow.feather as feather   # Package to store dataframes in a binary format
import joblib
import copy as cp

#Data manipulation
import pandas as pd
import numpy as np

#Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, make_pipeline
from more_itertools import powerset
from collections.abc import Iterable

#Hyperparameter tuning
import optuna  # Tuning the meta learner hyperparameters

#Base models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


#Possible future models
#from tensorflow import keras (ANN/LSTM)
#from xgboost import XGBClassifier
#from pgmpy.models import BayesianNetwork (BayesNet)

#Stacking models
from sklearn.ensemble import StackingClassifier

#Model evaluation
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_score

#Data visualisation
import matplotlib.pyplot as plt

#Setting working directory
os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")

#File paths to model artificats
intermediate = Path.cwd() / "data" / "intermediate"
#model_studies_location = Path.cwd() / "data" / "intermediate" / "model_studies_differentials"
model_studies_location = Path.cwd() / "data" / "intermediate" / "model_studies_differentials_leak_fix"

####################################
#Prepping Input Data
####################################

#Cross validation
tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)

#Loading in feature selected columns
#fs_columns = joblib.load(f"{intermediate}/fs_columns_differentials.pkl")
fs_columns = joblib.load(f"{intermediate}/fs_columns_differentials_leak_fix.pkl")

#Reading in the data that will be used in the model
#input_data = pd.read_feather(f"{intermediate}/input_data_differentials.feather")

#input_data = feather.read_feather(f"{intermediate}/all_football_data_differential.feather")
input_data = feather.read_feather(f"{intermediate}/all_football_data_differential_leak_fix.feather")
input_data = input_data.sort_values(by="date", ascending=True)
input_data.set_index("date", inplace=True)

#Sorting index to make sure that the data is in chronological order
input_data = input_data.sort_index()

#Reducing the data to only the feature selected columns and the result column (This is the target variable)
input_data_fs = input_data[fs_columns + ["result"]]


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
rf_study = joblib.load(f"{model_studies_location}/rf_study.pkl")
gb_study = joblib.load(f"{model_studies_location}/gb_study.pkl")
svc_study = joblib.load(f"{model_studies_location}/svc_study.pkl")
dimred_knn_study = joblib.load(f"{model_studies_location}/dimred_knn_study.pkl")
lr_study = joblib.load(f"{model_studies_location}/lr_study.pkl")
gnb_study = joblib.load(f"{model_studies_location}/gnb_study.pkl")

#Needing to make some adjustments to 

#KNN model
dimred_knn_study_best_params = dimred_knn_study.best_params.copy()

if dimred_knn_study_best_params["comp_analysis"] == "LDA":
    dimred_knn_method = LinearDiscriminantAnalysis(n_components=dimred_knn_study_best_params["lda_n_components"])
    del dimred_knn_study_best_params['comp_analysis']
    del dimred_knn_study_best_params['lda_n_components']
    
elif dimred_knn_study_best_params["comp_analysis"] == "PCA":
    dimred_knn_method = PCA(n_components=dimred_knn_study_best_params["pca_n_components"])
    del dimred_knn_study_best_params['comp_analysis']
    del dimred_knn_study_best_params['lda_n_components']
    
    
elif dimred_knn_study_best_params["comp_analysis"] == "NCA":
    dimred_knn_method = NeighborhoodComponentsAnalysis(n_components=dimred_knn_study_best_params["nca_n_components"])
    del dimred_knn_study_best_params['comp_analysis']
    del dimred_knn_study_best_params['lda_n_components']
else:
    dimred_knn_method = "passthrough"
    del dimred_knn_study_best_params['comp_analysis']
    del dimred_knn_study_best_params['lda_n_components']

#LR model
lr_study_best_params = lr_study.best_params.copy()

lr_study_best_params['solver'],lr_study_best_params['penalty'] = lr_study_best_params['solver_penalty'].split("_")
del lr_study_best_params['solver_penalty']


#Creating a dictionary of the base models that will be used in the ensemble
#Level 0 has been implemented as a dictionary of base classification models (This will be added to as more models are added to the ensemble)
level_0_classifiers = dict()
level_0_classifiers["rf"] = RandomForestClassifier(**rf_study.best_params, random_state=41)
level_0_classifiers["gb"] = GradientBoostingClassifier(**gb_study.best_params, random_state=41)
level_0_classifiers["svc"] = SVC(**svc_study.best_params, random_state=41) #Will put back in once I've retrained the model with the probability parameter
level_0_classifiers["dimred_knn"] = make_pipeline(dimred_knn_method, 
                                                  KNeighborsClassifier(**dimred_knn_study_best_params))
level_0_classifiers["lr"] = LogisticRegression(**lr_study_best_params, random_state=41)
level_0_classifiers["gnb"] = GaussianNB(**gnb_study.best_params)


#Using a random forest classifier as the level 1 classifier (This will eventually be hyperparameter tuned)
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


#Here we are going to hyperparameter tune the level 1 classifier. This is done using the Optuna library
#This is done using the Optuna library
def objective(trial):
        
        #Defining the hyperparameters that will be tuned
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        
        #Defining the model
        model = RandomForestClassifier(criterion=criterion, 
                                       n_estimators=n_estimators, 
                                       max_depth=max_depth, 
                                       min_samples_split=min_samples_split, 
                                       min_samples_leaf=min_samples_leaf,
                                       max_features=max_features, 
                                       bootstrap=bootstrap, 
                                       random_state=41)
        
        #Fitting the model
        model.fit(X_train_stacked.drop(labels=["index_number"], axis=1), y_train_stacked.drop(labels=["index_number"], axis=1))
        
        #Predicting the outcome of the test data
        y_pred_stacked = model.predict(X_train_stacked.drop(labels=["index_number"], axis=1))
        
        #Getting the accuracy score
        accuracy = accuracy_score(y_train_stacked, y_pred_stacked)
        
        return accuracy


#Fitting the level 1 classifier (meta-Learner) to the base input features + the stacked probabilities dataset (Should I be using some type of cross validation here?)
level_1_classifier.fit(X_train_stacked.drop(labels=["index_number"], axis=1), y_train_stacked.drop(labels=["index_number"], axis=1))

#Saving my level 1 stacked model
#joblib.dump(level_1_classifier, f"data/intermediate/level_1_classifier.pkl")
#joblib.dump(level_1_classifier, f"data/intermediate/level_1_classifier_differentials.pkl")
joblib.dump(level_1_classifier, f"data/intermediate/level_1_classifier_differentials_leak_fix.pkl")

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

#plotting the feature importance for the stacked model. These are sorted in order of importance
plt.figure(figsize=(10,10))

#Sorting the feature importances in order of importance
stacked_feature_importances = pd.Series(stacked_feature_importances, index=X_train_stacked.drop(labels=["index_number"], axis=1).columns).sort_values(ascending=True)
plt.barh(stacked_feature_importances.index, stacked_feature_importances)
plt.title("Feature Importance for Stacked Model")
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.show()


#Finally to see whether the stacked model was actually worth doing, I will compare the accuracy of the stacked model to the accuracy of the base models
print(f"Accuracy of scikit-learn stacking classifier: {accuracy_score(y_test, y_pred_stacked)}")

for name, classifier in level_0_classifiers.items():
    classifier_ = cp.deepcopy(classifier)
    classifier_.fit(X_train, y_train)

    print(f"Accuracy of standalone {name} classifier: {accuracy_score(y_test, classifier_.predict(X_test_scaled.drop(labels=['index_number'], axis=1)))}")

y_pred_stacked_df.value_counts(normalize=True)





#Now I am trying to find the optimal combination of models for the level 0 anbd level 1 classifiers (taken from https://towardsdatascience.com/a-deep-dive-into-stacking-ensemble-machine-learning-part-ii-69bfc0d6e53d)
#For this to work properly I need to create new clas or function that does what a stackingmodel does. SO encompassing all the stuff I have done above into a single model.


def power_set(items: Iterable, min_length : int = 0) -> list:
    list_of_tuples = list(powerset(items))
    list_of_lists = [list(elem) for elem in list_of_tuples]

    return [list for list in list_of_lists if len(list)>=min_length]

power_set(list(level_0_classifiers.keys()), 2)


param_grid = dict()
param_grid["estimators"] = power_set(list(level_0_classifiers.items()), 2)
param_grid["final_estimator"] = list(level_0_classifiers.values())
param_grid["passthrough"] = [True, False]
param_grid["stack_method"] = ["predict", "predict_proba"]

pre_defined_split = PredefinedSplit(test_fold = [-1 if x in X_train.index else 0 for x in X.index])
grid_search = GridSearchCV(estimator=stacking_model, param_grid=param_grid, scoring="accuracy", cv=tscv, verbose=10)
grid_search_results = grid_search.fit(X, y)








