"""
    The primary purpose of this script is to prepare the data and create features that the model will use to predict EPL results.
"""
# Imports
# Math and data manipulation imports
import os
from pathlib import Path

import pandas as pd  # Package for data manipulation and analysis
import numpy as np # Package for scientific computing
import lxml 
import html5lib
import io

# Web scraping imports
import requests  # Used to access and download information from websites
from bs4 import BeautifulSoup # Package for working with html and information parsed using requests
import time # Package to slow down the webscraping process

import pyarrow.feather as feather   # Package to store dataframes in a binary format

#Project directory locations
raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"


#TODO = Rolling team average XG (Expected Goals)
#TODO = 
#TODO = 
#TODO = add a feature to the data that is the number of days since the last time the user was active
#TODO = add a feature to the data that is the number of days since the last time the user was active
#TODO = add a feature to the data that is the number of days since the last time the user was active
#TODO = add a feature to the data that is the number of days since the last time the user was active
#TODO = add a feature to the data that is the number of days since the last time the user was active
#TODO = add a feature to the data that is the number of days since the last time  user was active
#TODO = 
#TODO = 
#TODO = a


#! = 

#Start by loading the match data,  elo data and team name lookup data
elo_ratings_all_teams = pd.read_feather(f"{intermediate}/elo_ratings_all_teams.feather")
match_data_all_teams  = pd.read_feather(f"{intermediate}/match_data_all_teams.feather")
team_name_lookup_df = pd.read_feather(f"{intermediate}/team_name_lookup_df.feather")

#Joining datasets together into a single combined dataframe (Can add more datasets to this later on)


#Merging the team_name_lookup_df to the match_df dataframe
match_data_all_teams = match_data_all_teams.merge(team_name_lookup_df, how="left", left_on="team", right_on="match_team_names")

#Now merging the elo_df_all_dates dataframe to the match_df dataframe
complete_data = match_data_all_teams.merge(elo_ratings_all_teams, how = "left", left_on=["elo_team_names", "date"], right_on=["club", "date"])


#Data Quality and Cleaning
#Checking for null values
complete_data.isnull().sum()

