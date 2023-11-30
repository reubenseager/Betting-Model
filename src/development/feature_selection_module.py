"""
    The purpose of the script is to look at which features to select for the model. This should not need to be ru each time, and so is not included in the primary processing.
"""
# Remove all objects from the global environment
for obj in list(globals()):
    del globals()[obj]

# Imports
# Math and data manipulation imports
import os
from pathlib import Path

import pandas as pd  # Package for data manipulation and analysis
import numpy as np # Package for scientific computing
import pytest # Package for testing code
from functools import reduce
from datetime import datetime
import pyarrow.feather as feather   # Package to store dataframes in a binary format

#Machine Learning Imports
#Train test split
from sklearn.model_selection import train_test_split

#Saling
from sklearn.preprocessing import StandardScaler

#Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import autofeatselect


os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")


#Project directory locations
raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"


####################################
#Feature Selection
####################################
#TODO = Look at using things like ANOVA to select the best features for the model.
#TODO = Look at using Genetic algorithms for feature selection (https://towardsdatascience.com/feature-selection-with-genetic-algorithms-7dd7e02dd237#:~:text=Genetic%20algorithms%20use%20an%20approach,model%20for%20the%20target%20task.)

#Reading in the data that will be used in the model

input_data = feather.read_feather(f"{intermediate}/all_football_data.feather")

#Set the date as the index. Also reorder the index so the most recent data is at the top. This is so that when we split the data into test and train sets, we don't use future data to predict past data.
input_data = input_data.sort_values(by="date", ascending=True)
input_data.set_index("date", inplace=True)


#Splitting the data into test and train datasets. This needs to be done before any feature selection is done to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(input_data.drop(columns=["result"], axis=1), input_data["result"], test_size=0.15, shuffle=False) #Setting shuffle to false, again to prevent data leakage

#scaling the numerical features of the input data. Need to make sure that I apply the same scaler to the test data s well to avoid data leakage. Also this is what the models will be built off
scaler = StandardScaler()


scaled_training_data = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)


#Scaling the input data but keeping the column names and index from the original dataframe
scaled_input_data= pd.DataFrame(scaler.fit_transform(input_data.drop(columns=["result"], axis=1)), 
                                columns=input_data.drop(columns=["result"], axis=1).columns, 
                                index=input_data.index)










# sklearn-genetic (https://pypi.org/project/sklearn-genetic/)
# https://sklearn-genetic.readthedocs.io/en/latest/api.html#genetic_selection.GeneticSelectionCV
# https://www.kaggle.com/code/tanmayunhale/genetic-algorithm-for-feature-selection
https://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination

#Can use random forest to select the best features
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE
https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00327-4
https://pgmpy.org/models/bayesiannetwork.html
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Fitting the anova feature selection to the data
anova_fs.fit(all_football_data.drop(columns=["date", "result"], axis=1), all_football_data["result"])

X_train_anova_fs = anova_fs.transform(all_football_data.drop(columns=["date", "result"], axis=1))

X_test_anova_fs = anova_fs.transform(all_football_data.drop(columns=["date", "result"], axis=1))
#Creating a dataframe of the anova feature selection sco
"""
For attribute selection the following at- tribute evaluators and search methods were used: CfsSubsetEval with BestFirst, ConsistencySubsetEval with BestFirst, WrapperSubsetEval (classifier: Naive- Bayes) with BestFirst

"""
