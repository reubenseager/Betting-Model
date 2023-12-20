"""
This script takes the pre-processed input data and trains multiple classfication models on said data.

These models will then be used in a stacked ensemble model to predict the final output. In order for the stacking algorithm to be effective, the models need to be as different as possible.
The models that I am planning to use are:
    - Random Forest (Done)
    - XGBoost (Or SKlearn's Gradient Boosting Classifier) (Done)
    - k-NN Classifier (Done)
    - SVM (Support Vector Machine) (Done)
    - Naive Bayes Network (PGMPY Implemntation)
    - Neural Network (Keras Implemntation)
    
Finally I am planning to use a logistic regression model as the meta learner to combine the outputs of the models mentioned above.
This model will then be stored so that it can be used later on. I also might want to retrain the model as I get new data as more games are played.

https://developer.ibm.com/articles/stack-machine-learning-models-get-better-results/ (This is good for model stacking)
http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/ (This is about cross validation stacking for classifiers)
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html#stacking-super-learning

https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
Apparently (according to the using bookmaker odds to predict football outcomes paper) the best model for draws are the ensemble-selection classifier
https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050922X00070/1-s2.0-S1877050922007955/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIC%2FcApy%2Bvpq6YwPPLg6VrhCy1nC9WYCvHJgyNcttRqrJAiEA%2FCpn08uWMJeqYBxEoFbnz%2Fi%2FpftxB8CSVwmsaNi4%2FpEqsgUIOBAFGgwwNTkwMDM1NDY4NjUiDNCNTMoYAT06cIhXaCqPBYVJGe1O5RPDqX0fEkW4sj3fZgJ8kSWbjFpHX%2FwPYS6%2FY7Asx0PV88HnhKG0F6A7hzZriVz1KBJkAweK%2FgHfn6pvBBnEw%2Bd5ThA2zD8iln1CvPW2QvjizwR5zr%2BIMHsVxRvkCovPGHDD9J%2FZK4x5%2BHJYm%2FzCzYV6Eqs1p3%2BiI4HZfoy5Na7cZym1v633FXZy4Kt6AGU6eSlgnJH1tv%2FX4GrdAY7%2BHKRc2SiQpuar88Re%2B%2BuOPZ9CPJR%2FmAXMEZC95R9O3uOP9J1T0RMcvQkKmWjsuEKMMViahc%2FnhMyOk7yNopxmXknXVVQU0%2FirVthaUKXaxDmka4Zss4vqbvA2q%2Fl6N7qJBJkaj8EImp1o8l5DovCSS5RSaWjDxXODoQjf5d6DQNuXDUmRL124m2Yjnfy2esJ4t1OYhnA44EdvAYPxL8inNn4Oqcay4Ixx%2Fb3sDfmrR2864v9dwt1OcuWVfJBFBuevTXeSzK9hJngQWnZXd58EmLWkkEVpIJGL6yK3OGbPii4mIUCoXBcwCyXfBoapeP%2FkWJSlekBcGf9qet1ocF48Asj2E7BtT3qpzaOn7KUyT7Ys8309xOVzAEaWH5t3tymZgqx4CFdDzjfQc98GnFBSCh7Yb0kp6477F1KJNNjRIxNqDR3uf7UNOg2QElR6gn5k3PBudWad4Kt5wrrTn85eM7x4MdLKsgXN3tMepkhbMVJEmj0O6s%2FfcRjRJlVys85OuKTZCaCgiQi5oEGlFFDD4ud6CwEvra531WLVDSs9TvNbztFET2wN6a%2FP9T4FRJJ6S6EBHODV5NvDkizMsl8ljp60n6W63aGPygwS%2FYAFGS83I%2BkEYfMb6zZOT7ukjgMi1vI%2BSxTOYuSvJp0w2crjqwY6sQHPU4aFj0Qu%2BYnyZjEKCa85KmOp98viI7x0OtD3px0uwZUOHCtb30ghfCFaqIF6RDHp2w2wgxyKfeuwaBX%2Ff3tPWDlncSS2s%2B0%2FTqeJ1EznsKpRraYSzB4U2C3i69hIuUG6njLkPxWkvAe2DgU4o5wUNr1fCvLRXGN0Wq9xSAfOxRNFobFRIZSR745HRpm9hOj0GTlJZ8cjZMmOct4SS8FkfodEeSQ2Tuo%2Fd520zUHql10%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231212T233822Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYUXTG3HF7%2F20231212%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=f3f48d8a9648998983c64a13a1b908ebcbb60c526285a61c13308ec46c7f9144&hash=89d255cf77dce50c5b927bec6ca2af213952c7c5a7e8ee13b527bb10b88865ca&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050922007955&tid=spdf-6fd6b4dc-89b9-4a03-aafc-40abd7908ee1&sid=7ad519851af06742f338d0b5e35843af061cgxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=05015d525258050d06&rr=8349cceeaccd76d2&cc=gb

https://www.kaggle.com/code/marketneutral/purged-time-series-cv-xgboost-optuna
Remember that once I have found the optimal hyperparameters for the models, I need to retrain the models on the entire dataset. This is because the hyperparameters are tuned to the training data and not the entire dataset.
"""
#Imports

#Directory management
import os
from pathlib import Path  
import pyarrow.feather as feather   # Package to store dataframes in a binary format
import joblib

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


#Project directory locations
raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"   

#Location to save the trained studies
Path("data/intermediate/model_studies").mkdir(exist_ok=True)

####################################
#Reading in the data
####################################

#Loading in feature selected columns
fs_columns = joblib.load(f"{intermediate}/fs_columns.save")

#Reading in the data that will be used in the model
input_data = pd.read_feather(f"{intermediate}/input_data.feather")

#Sorting index to make sure that the data is in chronological order
input_data = input_data.sort_index()

#Reducing the data to only the feature selected columns. W
input_data_fs = input_data[fs_columns + ["result"]]
#input_data_fs = input_data

#I will be using cross validation but I need to make sure that I hold out some test data for the final testing of the model. This is what I'm splitting out at this stage
#I'm holding back 10% of the data for final testing of the model
X_train, X_test, y_train, y_test = train_test_split(input_data_fs.drop(columns=["result"], axis=1), input_data_fs["result"], 
                                                    test_size=0.10, shuffle=False) #Setting shuffle to false, again to prevent data leakage

#Scaling the data. This scaler will also be applied to the test data
scaler = StandardScaler() #Creating the scaler object
scaler.fit(X_train) #Fitting the scaler object to the training data
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index = X_train.index)#scaling the training data



#Now I will be looking at using cross validation to tune the hyperparameters of the models
n_splits = 4
tscv = TimeSeriesSplit(n_splits=n_splits)

####################################
#Hyperparameter tuning of Base Models
####################################

####################
#Random Forest
####################

def rf_objective(trial):
    
    #Saving the study so it can be used later on
    #joblib.dump(study, f"{intermediate}/rf_study.pkl")
    
    #Parameters to tune
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_depth = trial.suggest_int("max_depth", 1, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    
    #creating the model
    rf = RandomForestClassifier(n_estimators=n_estimators, 
                                criterion=criterion, 
                                max_depth=max_depth, 
                                min_samples_split=min_samples_split, 
                                min_samples_leaf=min_samples_leaf, 
                                max_features=max_features, 
                                random_state=41)
    
    #Return time series cross validation f1 score
    f1_macro = cross_val_score(rf, X_train, y_train, cv=tscv, scoring="f1_macro").mean()
    return f1_macro

#Create study object
rf_study = optuna.create_study(direction="maximize", study_name="Random Forest Study")
rf_study.optimize(rf_objective, n_trials=200, show_progress_bar=True)

#Saving the study so it can be used later on
joblib.dump(rf_study, f"data/intermediate/model_studies/rf_study.pkl")
rf_study = joblib.load(f"data/intermediate/model_studies/rf_study.pkl")   #Loading in the study

#Retuning the best parameters
rf_best_params = rf_study.best_params

#returing the best score
rf_best_score = rf_study.best_value

####################
#Gradient Boosting
####################

def gb_objective(trial):

    #Parameters to tune
    loss = trial.suggest_categorical("loss", ["log_loss"])
    learning_rate = trial.suggest_float("learning_rate", 0.001, 1) #This shrinks the contribution of each tree
    n_estimators = trial.suggest_int("n_estimators", 50, 500) #This is the number of trees in the forest
    min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
    max_depth = trial.suggest_int("max_depth", 5, 50) #This is how deep the tree can be split
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])    

    #creating the model
    gb = GradientBoostingClassifier(loss=loss,
                                    learning_rate=learning_rate,
                                    n_estimators=n_estimators,
                                    min_samples_split=min_samples_split,
                                    max_depth=max_depth,
                                    max_features=max_features,
                                    random_state=41)

    
    #Return time series cross validation f1 score
    f1_macro = cross_val_score(gb, X_train, y_train, cv=tscv, scoring="f1_macro").mean()
    return f1_macro

#Create study object
gb_study = optuna.create_study(direction="maximize", study_name="Gradient Boosting Classifier Study")
gb_study.optimize(gb_objective, n_trials=200, show_progress_bar=True)

#Saving the study so it can be used later on
joblib.dump(gb_study, f"data/intermediate/model_studies/gb_study.pkl")
gb_study = joblib.load(f"data/intermediate/model_studies/gb_study.pkl")   #Loading in the study

#Retuning the best parameters
gb_best_params = gb_study.best_params

#Returing the best score
gb_best_score = gb_study.best_value

####################
#Support Vector Classifier
####################

def svc_objective(trial):

    #Parameters to tune
    C = trial.suggest_float("C", 0.001, 1000)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]) #Linear appears to be the best
    degree = trial.suggest_int("degree", 1, 10)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    shrinking = trial.suggest_categorical("shrinking", [True, False])
    probability = trial.suggest_categorical("probability", [False])
    class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    max_iter = trial.suggest_categorical("max_iter", [-1])
    decision_function_shape = trial.suggest_categorical("decision_function_shape", ["ovo", "ovr"])
 

    #creating the model
    svc = SVC(C=C,
              kernel=kernel,
              degree=degree,
              gamma=gamma,
              shrinking=shrinking,
              probability=probability,
              class_weight=class_weight,
              max_iter=max_iter,
              decision_function_shape=decision_function_shape,
              random_state=41)

    
    #Return time series cross validation f1 score
    f1_macro = cross_val_score(svc, X_train, y_train, cv=tscv, scoring="f1_macro").mean()
    return f1_macro

#Create study object
svc_study = optuna.create_study(direction="maximize", study_name="SVC study")
svc_study.optimize(svc_objective, n_trials=50, show_progress_bar=True) #Reducing the number of trials as this appears to be a very computationally expensive model

#Saving the study so it can be used later on
joblib.dump(svc_study, f"data/intermediate/model_studies/svc_study.pkl")
svc_study = joblib.load(f"data/intermediate/model_studies/svc_study.pkl")   #Loading in the study

#Retuning the best parameters
svc_best_params = svc_study.best_params

#Returing the best score
svc_best_score = svc_study.best_value

####################
#KNN Classifier
####################

def knn_objective(trial):
    
    #Adding a PCA step to the pipeline to see the effect of dimensionality reduction

    #Parameters to tune
    n_neighbors = trial.suggest_int("n_neighbors", 1, 100)
    weights = trial.suggest_categorical("weights", ["distance"])
    leaf_size = trial.suggest_int("leaf_size", 1, 100)
    p = trial.suggest_int("p", 1, 10)
    metric = trial.suggest_categorical("metric", ["minkowski"])
    
    #use_pca = trial.suggest_categorical("use_pca", [True, False])
    
    #Create the pipeline with hyperparameters
    # steps = [
    #     ('classifier', KNeighborsClassifier(n_neighbors=n_neighbors,
    #                                         weights=weights,
    #                                         leaf_size=leaf_size,
    #                                         p=p,
    #                                         metric=metric))
    # ]
    knn = KNeighborsClassifier(n_neighbors=n_neighbors,
                                            weights=weights,
                                            leaf_size=leaf_size,
                                            p=p,
                                            metric=metric)
    # if use_pca:
    #     n_components = trial.suggest_int('n_components', 1, min(X_train.shape))
    #     steps.insert(0, ('pca', PCA(n_components=n_components))) #Inserting the PCA step at the start of the pipeline
        
    #Creating the pipeline
    #knn = Pipeline(steps)
    
    #Return time series cross validation f1 score
    f1_macro = cross_val_score(knn, X_train, y_train, cv=tscv, scoring="f1_macro").mean()
    return f1_macro

#Create study object
knn_study = optuna.create_study(direction="maximize", study_name="kNN study")
knn_study.optimize(knn_objective, n_trials=500, show_progress_bar=True)

#Saving the study so it can be used later on
joblib.dump(knn_study, f"data/intermediate/model_studies/knn_study.pkl")

#Retuning the best parameters
knn_best_params = knn_study.best_params
knn_study = joblib.load(f"data/intermediate/model_studies/knn_study.pkl")   #Loading in the study

#Returing the best score
knn_best_score = knn_study.best_value

#fitting knn model using the best parameters
# knn_best_model = KNeighborsClassifier(**knn_best_params)
# knn_best_model.fit(X_train, y_train)

####################
#Naive Bayes Network
####################
def nbn_objective(trial):

    nbn = 

    
    #Return time series cross validation f1 score
    f1_macro = cross_val_score(nbn, X_train, y_train, cv=tscv, scoring="f1_macro").mean()
    return f1_macro

#Create study object
nbn_study = optuna.create_study(direction="maximize", study_name="nbn study")
nbn_study.optimize(knn_objective, n_trials=500, show_progress_bar=True)

#Saving the study so it can be used later on
joblib.dump(nbn_study, f"data/intermediate/model_studies/nbn_study.pkl")

#Retuning the best parameters
nbn_study_best_params = nbn_study.best_params
nbn_study = joblib.load(f"data/intermediate/model_studies/nbn_study.pkl")   #Loading in the study

####################################
#Training Base Models
####################################

#Each of the individual models will now be trained on the entire dataset using the best parameters found above (Maybe done using stacking classifier)
# rf_best_model.fit(X_train, y_train)
# gb_best_model.fit(X_train, y_train)
# svc_best_model.fit(X_train, y_train)
# knn_best_model.fit(X_train, y_train)
