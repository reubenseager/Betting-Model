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
    #year_odds = historical_betting_odds_files[1]

    #Reading in the Excel file
    betting_data = pd.read_excel(year_odds)
    #These are the columns that I am interested in. I will be using the average of the odds from all the bookmakers as the odds for the game.
    betting_cols = ["Date", "HomeTeam", "AwayTeam", 
                "B365H", "B365D", "B365A", 
                "BWH", "BWD", "BWA", 
                "IWH", "IWD", "IWA", 
                "PSH", "PSD", "PSA",
                "WHH", "WHD", "WHA", 
                "VCH", "VCD", "VCA"]

    #Reducing the columns down to only those betting columns that we are interested in.
    betting_data = betting_data[betting_cols]

    #Converting the date column to a datetime object
    betting_data["Date"] = pd.to_datetime(betting_data["Date"], format="%d/%m/%Y")

    #Selecting the home, away, and draw odds from each of the bookmakers and then taking the average of these odds to get the average odds for the game.
    home_odds_cols = ["B365H", "BWH", "IWH", "PSH", "WHH", "VCH"]
    away_odds_cols = ["B365A", "BWA", "IWA", "PSA", "WHA", "VCA"]
    draw_odds_cols = ["B365D", "BWD", "IWD", "PSD", "WHD", "VCD"]

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
all_historical_betting_data.to_feather(f"{intermediate}/all_historical_betting_data.feather") #Writing to a feather file in the intermediate folder

#Writing to excel file
#all_historical_betting_data.to_excel(f"{output}/all_historical_betting_data.xlsx", index=False)
#####################################
#Current betting odds
#####################################

#This will be done with an API most likely. Need betting odds from Bet365, WIlliam Hill, Pinnacle, VC Bet, Interwetten odds, Bet&Win odds. So I can recreate the historical betting odds features

api_key =""


