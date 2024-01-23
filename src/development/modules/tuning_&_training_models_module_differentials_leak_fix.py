"""
This script takes the pre-processed input data and trains multiple classfication models on said data.

These models will then be used in a stacked ensemble model to predict the final output. In order for the stacking algorithm to be effective, the models need to be as different as possible.
The models that I am planning to use are:
    - Random Forest (Done)
    - XGBoost (Or SKlearn's Gradient Boosting Classifier) (Done)
    - k-NN Classifier (Done)
    - SVM (Support Vector Machine) (Done)
    - Naive Bayes Network (PGMPY Implemntation)
    - CatBoost (Look into this)
    
Try also using some true probabilistic models such as:
    - Bayesian Network (PGMPY Implemntation)
    - Naive Bayes
    - Neural Network (Keras Implemntation)
    -LSTM (Think this would work well as it is a time series problem) (Seems to do very well for other people.)(https://github.com/krishnakartik1/LSTM-footballMatchWinner)

These models use a probabilitic frame work and work in a different way. This should help improve the performance of the stacked ensemble model as the models are more different.    
https://machinelearningmastery.com/probability-calibration-for-imbalanced-classification/
https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
    
Finally I am planning to use a final meta learner to combine the outputs of the models mentioned above.
This model will then be stored so that it can be used later on. I also might want to retrain the model as I get new data as more games are played.

In terms of splitting the data up I think I need three splits: (Not 100% sure on this)
    - Training data (80%) - This will be used to train the base learners using cross validation
    - Hold out data (10%) - This will be used to test the train the meta learner
    - Test data (10%) - This will be used to test the superlearner
    
Also reading that if you use the probability outputs rather than the pure classification outputs, you can improve your results. The probability outputs provide more context on the decisions beiung made (https://towardsdatascience.com/a-deep-dive-into-stacking-ensemble-machine-learning-part-i-10476b2ade3#:~:text=In%20a%20binary%20classification%20for,the%20stacking%20model%20improves%20significantly.)

https://developer.ibm.com/articles/stack-machine-learning-models-get-better-results/ (This is good for model stacking)
http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/ (This is about cross validation stacking for classifiers)
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html#stacking-super-learning

https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
Apparently (according to the using bookmaker odds to predict football outcomes paper) the best model for draws are the ensemble-selection classifier
https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050922X00070/1-s2.0-S1877050922007955/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIC%2FcApy%2Bvpq6YwPPLg6VrhCy1nC9WYCvHJgyNcttRqrJAiEA%2FCpn08uWMJeqYBxEoFbnz%2Fi%2FpftxB8CSVwmsaNi4%2FpEqsgUIOBAFGgwwNTkwMDM1NDY4NjUiDNCNTMoYAT06cIhXaCqPBYVJGe1O5RPDqX0fEkW4sj3fZgJ8kSWbjFpHX%2FwPYS6%2FY7Asx0PV88HnhKG0F6A7hzZriVz1KBJkAweK%2FgHfn6pvBBnEw%2Bd5ThA2zD8iln1CvPW2QvjizwR5zr%2BIMHsVxRvkCovPGHDD9J%2FZK4x5%2BHJYm%2FzCzYV6Eqs1p3%2BiI4HZfoy5Na7cZym1v633FXZy4Kt6AGU6eSlgnJH1tv%2FX4GrdAY7%2BHKRc2SiQpuar88Re%2B%2BuOPZ9CPJR%2FmAXMEZC95R9O3uOP9J1T0RMcvQkKmWjsuEKMMViahc%2FnhMyOk7yNopxmXknXVVQU0%2FirVthaUKXaxDmka4Zss4vqbvA2q%2Fl6N7qJBJkaj8EImp1o8l5DovCSS5RSaWjDxXODoQjf5d6DQNuXDUmRL124m2Yjnfy2esJ4t1OYhnA44EdvAYPxL8inNn4Oqcay4Ixx%2Fb3sDfmrR2864v9dwt1OcuWVfJBFBuevTXeSzK9hJngQWnZXd58EmLWkkEVpIJGL6yK3OGbPii4mIUCoXBcwCyXfBoapeP%2FkWJSlekBcGf9qet1ocF48Asj2E7BtT3qpzaOn7KUyT7Ys8309xOVzAEaWH5t3tymZgqx4CFdDzjfQc98GnFBSCh7Yb0kp6477F1KJNNjRIxNqDR3uf7UNOg2QElR6gn5k3PBudWad4Kt5wrrTn85eM7x4MdLKsgXN3tMepkhbMVJEmj0O6s%2FfcRjRJlVys85OuKTZCaCgiQi5oEGlFFDD4ud6CwEvra531WLVDSs9TvNbztFET2wN6a%2FP9T4FRJJ6S6EBHODV5NvDkizMsl8ljp60n6W63aGPygwS%2FYAFGS83I%2BkEYfMb6zZOT7ukjgMi1vI%2BSxTOYuSvJp0w2crjqwY6sQHPU4aFj0Qu%2BYnyZjEKCa85KmOp98viI7x0OtD3px0uwZUOHCtb30ghfCFaqIF6RDHp2w2wgxyKfeuwaBX%2Ff3tPWDlncSS2s%2B0%2FTqeJ1EznsKpRraYSzB4U2C3i69hIuUG6njLkPxWkvAe2DgU4o5wUNr1fCvLRXGN0Wq9xSAfOxRNFobFRIZSR745HRpm9hOj0GTlJZ8cjZMmOct4SS8FkfodEeSQ2Tuo%2Fd520zUHql10%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231212T233822Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYUXTG3HF7%2F20231212%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=f3f48d8a9648998983c64a13a1b908ebcbb60c526285a61c13308ec46c7f9144&hash=89d255cf77dce50c5b927bec6ca2af213952c7c5a7e8ee13b527bb10b88865ca&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050922007955&tid=spdf-6fd6b4dc-89b9-4a03-aafc-40abd7908ee1&sid=7ad519851af06742f338d0b5e35843af061cgxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=05015d525258050d06&rr=8349cceeaccd76d2&cc=gb

https://www.kaggle.com/code/marketneutral/purged-time-series-cv-xgboost-optuna
Remember that once I have found the optimal hyperparameters for the models, I need to retrain the models on the entire dataset. This is because the hyperparameters are tuned to the training data and not the entire dataset.

#Possibly look at upsampling the draw outcomes as this is the worst performing outcome.
 Eryarsoy and Delen (2019) was the first study to address the poor
performance on draw outcomes by applying SMOTE (Chawla et al., 2002).
https://arno.uvt.nl/show.cgi?fid=160932

"""
#Imports

#Directory management
import os
from pathlib import Path  
import pyarrow.feather as feather   # Package to store dataframes in a binary format
import joblib

#Model preparation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline

#Base models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from pgmpy.models import BayesianNetwork
from tensorflow import keras

#LSTM
#Stacking models
from sklearn.ensemble import StackingClassifier

#Hyperparameter tuning
import optuna   #You can get a progress bar for optuna

#Visualisation
import matplotlib.pyplot as plt
import seaborn as sns



#Model evaluation
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_score

#Setting the working directory
os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")


#Project directory locations
raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"   

#Location to save the trained studies (New folder for the differentials studies)
#Path("data/intermediate/model_studies").mkdir(exist_ok=True)
model_studies_location = Path.cwd() / "data" / "intermediate" / "model_studies_differentials_leak_fix"
model_studies_location.mkdir(exist_ok=True)
retrain_models = True #This is a flag to determine whether the models should be retrained or not. If this is set to false then the models will be loaded in from the model folder

####################################
#Reading in the data
####################################§

#Loading in feature selected columns
#fs_columns = joblib.load(f"{intermediate}/fs_columns_differentials.pkl")
fs_columns = joblib.load(f"{intermediate}/fs_columns_differentials_leak_fix.pkl")

#Reading in the data that will be used in the model
#input_data = feather.read_feather(f"{intermediate}/all_football_data_differential.feather")
input_data = feather.read_feather(f"{intermediate}/all_football_data_differential_leak_fix.feather")

input_data = input_data.sort_values(by="date", ascending=True)
input_data.set_index("date", inplace=True)


#Reducing the data to only the feature selected columns. W
#input_data_fs = input_data[fs_columns + ["result"]]
input_data_fs = input_data

#dropping the ip_ columns as I want to see how thwe model does without them
#input_data_fs.drop(columns=[col for col in input_data_fs.columns if "ip_" in col], inplace=True)

#only selecting the differentials columns
#input_data_fs = input_data_fs[[col for col in input_data_fs.columns if "_diff" in col] + ["result"]]

#Just betting odds (0.439254) not scaled(0.439728)
#input_data_fs = input_data_fs[[col for col in input_data_fs.columns if col.startswith("ip_")] + ["result"]]


#Differential + betting odds
input_data_fs = input_data_fs[[col for col in input_data_fs.columns if col.endswith("_differential") or col.startswith("ip_")] + ["result"]]


#Seeing how the model does wihtout these features as I can get much more data if I drop them
input_data_fs.drop(columns = ['xg_rolling_differential', 
                              'xga_rolling_differential',
                              'npxg_rolling_differential',
                              'dist_rolling_differential'], inplace=True)

#I will be using cross validation but I need to make sure that I hold out some test data for the final testing of the model. This is what I'm splitting out at this stage
#I'm holding back 10% of the data for final testing of the model
X_train, X_test, y_train, y_test = train_test_split(input_data_fs.drop(columns=["result"], axis=1), input_data_fs["result"], 
                                                    test_size=0.15, shuffle=False) #Setting shuffle to false, again to prevent data leakage
#testing with less data


#Scaling the data. This scaler will also be applied to the test data
scaler = StandardScaler() #Creating the scaler object

#joblib.dump(scaler, f"{intermediate}/scaler.pkl") #Saving the scaler object so it can be used on the test data

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

if retrain_models == True:
    #Create study object
    rf_study = optuna.create_study(direction="maximize", study_name="Random Forest Study")
    rf_study.optimize(rf_objective, n_trials=200, show_progress_bar=True)
    
    #Saving the study so it can be used later on
    joblib.dump(rf_study, f"{model_studies_location}/rf_study.pkl")
else:
    rf_study = joblib.load(f"{model_studies_location}/rf_study.pkl")

#Retuning the best parameters
rf_best_params = rf_study.best_params

#returing the best score
rf_best_score = rf_study.best_value

rf_study_cheat = joblib.load(f"data/intermediate/model_studies_differentials_leak_fix first run/rf_study.pkl")

rf_study_cheat_best_score = rf_study_cheat.best_value

# rf_study_orig = joblib.load(f"data/intermediate/model_studies/rf_study.pkl")
# rf_study_orig_best_score = rf_study_orig.best_value

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

if retrain_models == True:
    #Create study object
    gb_study = optuna.create_study(direction="maximize", study_name="Gradient Boosting Classifier Study")
    gb_study.optimize(gb_objective, n_trials=200, show_progress_bar=True)
    
    #Saving the study so it can be used later on
    joblib.dump(gb_study, f"{model_studies_location}/gb_study.pkl")
else:
    gb_study = joblib.load(f"{model_studies_location}/gb_study.pkl")

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
    probability = trial.suggest_categorical("probability", [True])#Changed to true as this is needed for the stacking classifier
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

if retrain_models == True:
    #Create study object
    svc_study = optuna.create_study(direction="maximize", study_name="SVC study")
    svc_study.optimize(svc_objective, n_trials=50, show_progress_bar=True) #Reducing the number of trials as this appears to be a very computationally expensive model
    
    #Saving the study so it can be used later on
    joblib.dump(svc_study, f"{model_studies_location}/svc_study.pkl")
else:
    svc_study = joblib.load(f"{model_studies_location}/svc_study.pkl")
    
#Retuning the best parameters
svc_best_params = svc_study.best_params
#Returing the best score
svc_best_score = svc_study.best_value

####################
#Dim Reduction + KNN Classifier
####################
#(https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_dim_reduction.html)
#https://www.kaggle.com/discussions/general/271613

def dimred_knn_objective(trial):
    #Parameters to tune
    n_neighbors = trial.suggest_int("n_neighbors", 1, 100)
    weights = trial.suggest_categorical("weights", ["distance"])
    leaf_size = trial.suggest_int("leaf_size", 1, 100)
    p = trial.suggest_int("p", 1, 10)
    metric = trial.suggest_categorical("metric", ["minkowski"]) #Should probably look at more distance metrics here
    dim_red = trial.suggest_categorical("comp_analysis", ["PCA", "LDA", "NCA", None])
        
    #Dimensionality reduction
    if dim_red == "PCA":
        n_components=trial.suggest_int("pca_n_components", 1, min(X_train.shape)) # suggest an integer from 2 to 30
        dimen_red_algorithm=PCA(n_components=n_components, random_state=41)
    elif dim_red == "LDA":
        n_components=trial.suggest_int("lda_n_components", 1, 2)
        dimen_red_algorithm=LinearDiscriminantAnalysis(n_components=n_components)
    elif dim_red == "NCA":
        n_components=trial.suggest_int("nca_n_components", 1, min(X_train.shape))
        dimen_red_algorithm=NeighborhoodComponentsAnalysis(n_components=n_components, random_state=41)
    else:
        dimen_red_algorithm='passthrough'
        
    knn = KNeighborsClassifier(n_neighbors=n_neighbors,
                                        weights=weights,
                                        leaf_size=leaf_size,
                                        p=p,
                                        metric=metric)
    
    # -- Make a pipeline
    pipeline = make_pipeline(dimen_red_algorithm, knn)
    
    #Return time series cross validation f1 score
    f1_macro = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring="f1_macro").mean()
    return f1_macro

if retrain_models == True:
    #Create study object
    dimred_knn_study = optuna.create_study(direction="maximize", study_name="dimred kNN study")
    dimred_knn_study.optimize(dimred_knn_objective, n_trials=250, show_progress_bar=True)
    
    #Saving the study so it can be used later on
    joblib.dump(dimred_knn_study, f"{model_studies_location}/dimred_knn_study.pkl")
else:
    dimred_knn_study = joblib.load(f"{model_studies_location}/dimred_knn_study.pkl")

#Returning the best parameters
dimred_knn_best_params = dimred_knn_study.best_params
#Returning the best score
dimred_knn_best_score = dimred_knn_study.best_value


# knn_study_cheat = joblib.load(f"data/intermediate/model_studies_differentials_leak_fix first run/dimred_knn_study.pkl")
# knn_study_cheat = joblib.load(f"data/intermediate/model_studies_differentials/dimred_knn_study.pkl")

# knn_study_cheat_best_score = knn_study_cheat.best_value


####################
#Naive Bayes (Gaussian)
####################


#checking for correlation between the features


plt.figure(figsize=(12,10))
cor = X_train.corr().abs()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Trying dimensioanlty reduction before naive bayes to see if it improves the model
def dimred_gnb_objective(trial):
    
    #Parameters to tune
    var_smoothing = trial.suggest_float("var_smoothing", 0.000000001, 0.0000001)
    dim_red = trial.suggest_categorical("comp_analysis", ["PCA", None])
        
    #Dimensionality reduction
    if dim_red == "PCA":
        n_components=trial.suggest_int("pca_n_components", 1, min(X_train.shape)) # suggest an integer from 2 to 30
        dimen_red_algorithm=PCA(n_components=n_components, random_state=41)
    else:
        dimen_red_algorithm='passthrough'
    
   
    gnb = GaussianNB(var_smoothing=var_smoothing)
    
        # -- Make a pipeline
    pipeline = make_pipeline(dimen_red_algorithm, gnb)
    #Return time series cross validation f1 score
    
    #Trialing both the standard and reduced datasets
        
    f1_macro = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring="f1_macro").mean()
    return f1_macro

if retrain_models == True:
    #Create study object
    dimred_gnb_study = optuna.create_study(direction="maximize", study_name="gnb study")
    dimred_gnb_study.optimize(dimred_gnb_objective, n_trials=200, show_progress_bar=True)
    
    #Saving the study so it can be used later on
    joblib.dump(dimred_gnb_study, f"{model_studies_location}/dimred_gnb_study.pkl")
else:
    dimred_gnb_study = joblib.load(f"{model_studies_location}/dimred_gnb_study.pkl")
    
#Returning the best parameters
dimred_gnb_study_best_params = dimred_gnb_study.best_params
#Returning the best score
dimred_gnb_study_best_score = dimred_gnb_study.best_value


def gnb_objective(trial):
    
    #This is the Gaussian Naive Bayes implementation
    
    #Parameters to tune
    var_smoothing = trial.suggest_float("var_smoothing", 0.000000001, 0.0000001)
    
    gnb = GaussianNB(var_smoothing=var_smoothing)
    
    #Return time series cross validation f1 score
    
    #Trialing both the standard and reduced datasets
        
    f1_macro = cross_val_score(gnb, X_train, y_train, cv=tscv, scoring="f1_macro").mean()
    return f1_macro

if retrain_models == True:
    #Create study object
    gnb_study = optuna.create_study(direction="maximize", study_name="dimred gnb study")
    gnb_study.optimize(gnb_objective, n_trials=200, show_progress_bar=True)
    
    #Saving the study so it can be used later on
    joblib.dump(gnb_study, f"{model_studies_location}/gnb_study.pkl")
else:
    gnb_study = joblib.load(f"{model_studies_location}/gnb_study.pkl")
    
#Returning the best parameters
gnb_study_best_params = gnb_study.best_params
#Returning the best score
gnb_study_best_score = gnb_study.best_value

# gnb_study_cheat = joblib.load(f"data/intermediate/model_studies_differentials/dimred_knn_study.pkl")

# gnb_study_cheat_best_score = gnb_study_cheat.best_value

####################
#Naive Bayes (Complement) (IF I WANNA USE THIS THEN I NEED TO CHANGE THE DATA TO BE POSITIVE. USE MINMAX SCALER)
####################
# def cnb_objective(trial):
    
#     #This is the Complement Naive Bayes implementation
    
#     #Parameters to tune
#     alpha = trial.suggest_float("alpha", 0.0, 1.0)
#     fit_prior = trial.suggest_categorical("fit_prior", [True, False])
#     norm = trial.suggest_categorical("norm", [True, False])
        
#     cnb = ComplementNB(alpha=alpha, 
#                        fit_prior=fit_prior, 
#                        norm=norm)
    
#     #Return time series cross validation f1 score
#     f1_macro = cross_val_score(cnb, X_train, y_train, cv=tscv, scoring="f1_macro").mean()
#     return f1_macro

# #Create study object
# cnb_study = optuna.create_study(direction="maximize", study_name="cnb study")
# cnb_study.optimize(cnb_objective, n_trials=200, show_progress_bar=True)

# #Saving the study so it can be used later on
# joblib.dump(cnb_study, f"data/intermediate/model_studies_differentials/cnb_study.pkl")

# #Retuning the best parameters
# cnb_study_best_params = cnb_study.best_params

####################
#Logistic Regression
####################
def lr_objective(trial):
    
    # Define all possible combinations of solver and penalty
    solver_penalty_combinations = [
        "newton-cg_l2", 
        #"newton-cg_None", 
        "lbfgs_l2", 
        #"lbfgs_None", 
        "liblinear_l1", 
        "liblinear_l2", 
        "sag_l2", 
        #"sag_None", 
        "saga_l1", 
        "saga_l2", 
        "saga_elasticnet", 
        #"saga_None"
    ]

    # Suggest a combination
    solver_penalty = trial.suggest_categorical("solver_penalty", solver_penalty_combinations)

    # Split the string back into separate variables
    solver, penalty = solver_penalty.split("_")
        
        
    l1_ratio =  None

    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

    # Other parameters...
    C = trial.suggest_float("C", 0.001, 1000)
    #l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    max_iter = trial.suggest_int("max_iter", 50, 500000)
    class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    multi_class = trial.suggest_categorical("multi_class", ["auto"])
    
    lr = LogisticRegression(penalty=penalty,
                            C=C,
                            solver=solver,
                            l1_ratio=l1_ratio,
                            max_iter=max_iter,
                            class_weight=class_weight,
                            multi_class=multi_class,
                            random_state=41)
    
    
    #Return time series cross validation f1 score
    f1_macro = cross_val_score(lr, X_train, y_train, cv=tscv, scoring="f1_macro").mean()
    return f1_macro

if retrain_models == True:
    #Create study object
    lr_study = optuna.create_study(direction="maximize", study_name="lr study")
    lr_study.optimize(lr_objective, n_trials=200, show_progress_bar=True)
    
    #Saving the study so it can be used later on
    joblib.dump(lr_study, f"{model_studies_location}/lr_study.pkl")
else:
    lr_study = joblib.load(f"{model_studies_location}/lr_study.pkl")

#Retuning the best parameters
lr_study_best_params = lr_study.best_params
#Returning the best score
lr_best_score = lr_study.best_value

####################
#Naive Bayes Network
####################

####################
#Multi Layer Perceptron
####################
def mlp_study:
    #https://www.kaggle.com/ryanholbrook/optimizing-mlp-hyperparameters-with-optuna
    #



####################
#LSTM
####################