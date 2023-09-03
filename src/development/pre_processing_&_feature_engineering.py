"""
    The primary purpose of this script is to prepare the data and create features that the model will use to predict EPL results.
"""
# Imports
# Math and data manipulation imports
import os
from pathlib import Path

import pandas as pd  # Package for data manipulation and analysis
import numpy as np # Package for scientific computing
import pytest # Package for testing code

# Web scraping imports
import requests  # Used to access and download information from websites
from bs4 import BeautifulSoup # Package for working with html and information parsed using requests
import time # Package to slow down the webscraping process
import lxml 
import html5lib
import io

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
#TODO = Shots conceded last 5 games
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
#complete_data = complete_data.drop(columns=["match_team_names", "elo_team_names", "club", "gls", "index"], axis=1,   errors="ignore")

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

#Writing a test to check that there are no missing values in the data. Print out which columns are missing values if there are any (NOT sure if this works or how you use pytest)
def test_no_missing_values():
    assert complete_data.isnull().sum().sum() == 0, "There are missing values in the data"
    
test_no_missing_values()
  

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


def rolling_averages(group, cols, new_cols, window):
    #Start by sorting the data by date as we are looking at recent form
    group = group.sort_values(by="date")
    
    #Closed = left means that the window will ignore the result of the current game. As we do not want to include future informaion in the model
    rolling_stats = group[cols].rolling(window = window, closed = "left").mean()
    group[new_cols] = rolling_stats
    group = group.dropna()
    return group

cols_for_rolling = ["gf", "ga", "xg", "xga", "npxg" ,"points", "sh", "poss", "sot", "dist"]
new_rolling_cols = [f"{col}_rolling" for col in cols_for_rolling]


complete_data_rolling = complete_data.groupby("team").apply(lambda x: rolling_averages(x, cols=cols_for_rolling, new_cols=new_rolling_cols, window=5))

#Write code that calculates the rolling averages for each team for the last 5 games for the different venue code types

#Creating some information relating to the difference in form between home and away results
home_away = complete_data[["date", "team", "venue", "points"]]
home_away = home_away.sort_values(by="date")

#Create columns called home_ppg and away_ppg that will be the rolling averages of the points column for the last 5 games for each team at home and away. Doing smaller window as games are less frequent
home_away_rolling_ppg = home_away.groupby(["team", "venue"])["points"].rolling(window=3, closed="left").mean()

#Converting the home_ppg series to a dataframe
home_away_rolling_ppg = pd.DataFrame(home_away_rolling_ppg).reset_index(level=["team", "venue"])

home_away_rolling_ppg.rename(columns={"points": "home_away_ppg_rolling"}, inplace=True)

#Joining the home_away_rolling_ppg dataframe to the home_away dataframe
home_away_rolling_ppg = home_away.merge(home_away_rolling_ppg[["home_away_ppg_rolling"]], how="left", left_index=True, right_index=True)

#Merging the home away rolling ppg dataframe to the complete_data dataframe
#complete_data_rolling = complete_data_rolling.merge(home_away_rolling_ppg[["home_away_ppg_rolling"]], how="left", left_index=True, right_index=True)

#Performance against previous club
#Here I will be looking at how well the team has done against the previous club over their last 5 max butwill allow just previous fixture fixtures


#Dropping the extra index level
complete_data_rolling = complete_data_rolling.droplevel("team")

#Resetting and then re-assigning the index to ensure we have unique values
complete_data_rolling.reset_index(drop=True)

#Reassignign the index values as we want to ensure we have unique values
complete_data_rolling.index = range(len(complete_data_rolling))

#Dropping the non-rolling averaged versions of the rolling columns
complete_data_rolling = complete_data_rolling.drop(columns=cols_for_rolling, axis=1)




#Merging all the datasets into a single combined dataset