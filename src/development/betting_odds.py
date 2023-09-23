"""
This script is looking at extracting betting odds/lines from the web. EIther via web scraping or API calls.
This data will then be input into the model ot see if it can improve the predictive output of the model.


The historical odds will most likely be extracted from Excel sheets.

#TODO: Maybe do some type of spread on the betting odds to and see if that helps the model predict results better.
#TODO: Maybe create a single name lookup table that all the team names can match to. So that I can use the same team names across all the datasets.

Think I'll basically have a few numbers. I think It would be good to just have the teams odds as a column (the opponenets odds and the draw odds. )
"""

#Imports
import os
from pathlib import Path  
import pyarrow.feather as feather   # Package to store dataframes in a binary format
import requests

import pandas as pd
import glob

os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")


#Project directory locations
raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"   

#####################################
#Historical Betting odds
#####################################

#From general betting knowledge, I know that Pinacle Sports is the "sharpest" good bookmaker.

#Listing all the historical betting odds in the raw folder
historical_betting_odds_files = glob.glob(f"{raw}/historical_betting_odds/betting_odds*.xlsx")
historical_betting_odds_files.sort(reverse=True) #Sorting the files in reverse order so that the most recent file is first in the list


def create_betting_data(year_odds):
    #Testing
    #year_odds = historical_betting_odds_files[1]

    #Reading in the Excel file
    betting_data = pd.read_excel(year_odds)
    #These are the columns that I am interested in. I will be using the average of the odds from all the bookmakers as the odds for the game.
    # betting_cols = ["Date", "HomeTeam", "AwayTeam", 
    #             "B365H", "B365D", "B365A", 
    #             "BWH", "BWD", "BWA", 
    #             "IWH", "IWD", "IWA", 
    #             "PSH", "PSD", "PSA",
    #             "WHH", "WHD", "WHA", 
    #             "VCH", "VCD", "VCA"]
    
    #Changed which bookmakers I am looking at to match up to what I can get from the API
    betting_cols = ["Date", "HomeTeam", "AwayTeam", 
                #"LBH", "LBD", "LBA", #Ladbrokes
                "PSH", "PSD", "PSA", #Pinnacle
                "WHH", "WHD", "WHA"] #William Hill

    #Reducing the columns down to only those betting columns that we are interested in.
    betting_data = betting_data[betting_cols]

    #Converting the date column to a datetime object
    betting_data["Date"] = pd.to_datetime(betting_data["Date"], format="%d/%m/%Y")

    #Selecting the home, away, and draw odds from each of the bookmakers and then taking the average of these odds to get the average odds for the game.
    home_odds_cols = ["PSH", "WHH"]
    away_odds_cols = ["PSA", "WHA"]
    draw_odds_cols = ["PSD", "WHD"]

    #Caclulating the average odds for each game
    betting_data["home_odds_avg"] = betting_data[home_odds_cols].mean(axis=1)
    betting_data["away_odds_avg"] = betting_data[away_odds_cols].mean(axis=1)
    betting_data["draw_odds_avg"] = betting_data[draw_odds_cols].mean(axis=1)

    #Reducing the dataframe down to only the columns that we are interested in
    final_betting_cols = ["Date", "HomeTeam", "AwayTeam", "home_odds_avg", "away_odds_avg", "draw_odds_avg", "PSH", "PSA", "PSD"]
    betting_data= betting_data[final_betting_cols]

    return betting_data
    
    
list_of_dfs = [create_betting_data(year_odds) for year_odds in historical_betting_odds_files]

    
    
all_historical_betting_data = pd.concat(list_of_dfs, axis=0) #Axis=0 to specify that we're concatenating row-wise
 
#Resetting the index
all_historical_betting_data = all_historical_betting_data.reset_index(drop=True)

#Converting Date column to datetime object
all_historical_betting_data["Date"] = pd.to_datetime(all_historical_betting_data["Date"], format="%d/%m/%Y")

#Renaming the columns to match the naming convention of the API
all_historical_betting_data = all_historical_betting_data.rename(columns={"Date": "date", 
                                                                          "HomeTeam": "home_team", 
                                                                          "AwayTeam": "away_team", 
                                                                          "home_odds_avg": "odds_h_avg", 
                                                                          "away_odds_avg": "odds_a_avg", 
                                                                          "draw_odds_avg": "odds_d_avg", 
                                                                          "PSH": "odds_h_pinnacle", 
                                                                          "PSA": "odds_a_pinnacle", 
                                                                          "PSD": "odds_d_pinnacle"})



all_historical_betting_data.to_feather(f"{intermediate}/all_historical_betting_data.feather") #Writing to a feather file in the intermediate folder


#Renaming the team names to the full team names that all the other datasets use
team_name_lookup = pd.read_excel(f"{raw}/team_name_lookup.xlsx")

#Merging the team name lookup table with the betting data
all_historical_betting_data = pd.merge(all_historical_betting_data, team_name_lookup, left_on="home_team", right_on="alternate_name", how="left")
all_historical_betting_data = all_historical_betting_data.drop(columns=["alternate_name"]).rename(columns={"correct_name": "home_team_full_name"})

all_historical_betting_data = pd.merge(all_historical_betting_data, team_name_lookup, left_on="away_team", right_on="alternate_name", how="left")
all_historical_betting_data = all_historical_betting_data.drop(columns=["alternate_name"]).rename(columns={"correct_name": "away_team_full_name"})

#Drop duplicate rows (May be a better way of doing the merge so that I don't have to do this)
all_historical_betting_data = all_historical_betting_data.drop_duplicates()


#Dropping the home_team and away_team columns
all_historical_betting_data = all_historical_betting_data.drop(columns=["home_team", "away_team"])

#Reordering the columns
all_historical_betting_data = all_historical_betting_data[["date", "home_team_full_name", "away_team_full_name", 
                                                           "odds_h_avg", "odds_a_avg", "odds_d_avg", 
                                                           "odds_h_pinnacle", "odds_a_pinnacle", "odds_d_pinnacle"]]


all_historical_betting_data.reset_index(drop=True, inplace=True)
#Writing to a feather file in the intermediate folder
all_historical_betting_data.to_feather(f"{intermediate}/all_historical_betting_data.feather")


#Writing to excel file so I can haev a look at the data more easily in Excel
all_historical_betting_data.to_excel(f"{output}/all_historical_betting_data.xlsx", index=False)

