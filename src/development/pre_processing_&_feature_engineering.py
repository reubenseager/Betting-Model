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

os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")


#Project directory locations
raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"


#TODO = Rolling team average XG (Expected Goals)
#TODO = Do I wanna keep the time of match variable. (Think probably not)
#TODO = Need to keep a time from last game variable 
#TODO = squad value (https://www.footballtransfers.com/en/values/teams/most-valuable-teams)
#TODO = Figure out if cat.codes gives the same codes to the same teams if we recreate the model next week. Maybe look ayt using sklearn encoders
#TODO = Filter to occasions where both the home and away prediction (of the same game) are the same. 
#TODO = average percentage of stadium capacity filled
#TODO = Sentiment analysis of twitter data (https://www.scraperapi.com/resources/)
#TODO = 
#TODO = 
#TODO = a


#! = 

#The web scraping and ELO ratings scripts shouldn't need to be run more than once. I should be able to work from this script only.

#Start by loading the match data,  elo data and team name lookup data
elo_ratings_all_teams = pd.read_feather(f"{intermediate}/elo_ratings_all_teams.feather")
match_data_all_teams  = pd.read_feather(f"{intermediate}/match_data_all_teams.feather")
team_name_lookup_df = pd.read_feather(f"{intermediate}/team_name_lookup_df.feather")

#Joining datasets together into a single combined dataframe (Can add more datasets to this later on)


#Merging the team_name_lookup_df to the match_df dataframe
match_data_all_teams = match_data_all_teams.merge(team_name_lookup_df, how="left", left_on="team", right_on="match_team_names")

#Now merging the elo_df_all_dates dataframe to the match_df dataframe
complete_data = match_data_all_teams.merge(elo_ratings_all_teams, how = "left", left_on=["elo_team_names", "date"], right_on=["club", "date"])

#Dropping the excess name columns
complete_data = complete_data.drop(columns=["match_team_names", "elo_team_names", "club", "gls"], axis=1,   errors="ignore")

#Looking at the shape of the data
complete_data.shape

#Seeing how many games there are per team using value counts
complete_data["team"].value_counts()


#Data Cleaning & Feature Engineering

#SHould probably do some checks for null values in the data. Should also probably do some plots to see if there are any outliers in the data
complete_data.isnull().sum()

#Do some type of pytest here to check that the data is in the correct format

#If dist is missing and has a nan value, then fill it with the highest value for that team. This is because dist being nan means the team has had no shots. Which should be punished by the model
complete_data["dist"] = complete_data.groupby("team")["dist"].transform(lambda x: x.fillna(x.max()))

#Again checking whether there are any missing values in the data
complete_data.isnull().sum()


#Checking data types. Need numeric data as that is what ML models take as input
complete_data.dtypes

#Dropping some more columns. Need to reevaluate this later on
complete_data = complete_data.drop(columns=["comp","day"], axis=1,   errors="ignore")

#Converting venue into a numeric coded variable
complete_data["venue"] = complete_data["venue"].astype("category")
complete_data["venue_code"] = complete_data["venue"].cat.codes

#Creating a code for the opponent column
complete_data["opponent"] = complete_data["opponent"].astype("category")
complete_data["opponent_code"] = complete_data["opponent"].cat.codes

#Creating a points column where a W is 3 points, a D is 1 point and a L is 0 points
complete_data["points"] = complete_data["result"].map({"W":3, "D":1, "L":0})

#Function that will create a rolling average of the data for the last 5 games for each team
def rolling_average(df, column, window):
    return df.groupby("team")[column].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)

#Creating a list of the columns that I want to create rolling averages for
cols_to_average = ["gf", "ga", "xg", "xga", "npxg" ,"points", "sh", "poss", "sot", "dist"]

#Creating new dataframe containing the rolling averages of the columns specifed int cols_to_average
rolling_averages = [rolling_average(complete_data, col, 5) for col in cols_to_average]





#Data Quality and Cleaning
#Checking for null values
complete_data.isnull().sum()

