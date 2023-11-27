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
data = feather.read_feather(f"{intermediate}/all_football_data.feather")


#sklearn-genetic (https://pypi.org/project/sklearn-genetic/)
#https://sklearn-genetic.readthedocs.io/en/latest/api.html#genetic_selection.GeneticSelectionCV
#https://www.kaggle.com/code/tanmayunhale/genetic-algorithm-for-feature-selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#Anova feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

#Selecting all features
anova_fs = SelectKBest(score_func=f_classif, k='all')

#Fitting the anova feature selection to the data
anova_fs.fit(all_football_data.drop(columns=["date", "result"], axis=1), all_football_data["result"])

X_train_anova_fs = anova_fs.transform(all_football_data.drop(columns=["date", "result"], axis=1))

X_test_anova_fs = anova_fs.transform(all_football_data.drop(columns=["date", "result"], axis=1))
#Creating a dataframe of the anova feature selection sco
"""
For attribute selection the following at- tribute evaluators and search methods were used: CfsSubsetEval with BestFirst, ConsistencySubsetEval with BestFirst, WrapperSubsetEval (classifier: Naive- Bayes) with BestFirst

"""
