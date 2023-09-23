import requests  # Used to access and download information from websites
import json  # Used to convert json data into python dictionaries
from functools import reduce  # Used to merge multiple dataframes together
import pandas as pd

import os
import glob
from pathlib import Path  

os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")


#Project directory locations
raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"   


#TODO= May need to create the inverse of the games so I have the data for both teams in the game
#TODO= Possibly look into somthing where I caulculate the chnage in odds over the last few dasys in the run up to the game. This can give some idea about the confidence in the teams. (May be a procy for injuries etc)

#NOTE: This needs to be run before the matches hgave started. Otherwise the dates will be off and some odds won't get uploaded.

#This is a free to access subscription levels to the odds api, which allows me to make up to 500 requests per month (https://api.the-odds-api.com/)

#My personal API key. (Emailed to me when I signed up to the odds api)
API_KEY = '2884dc7f2d10772d3e86f853ba262963'

#Only interested in the English Premier League at the moment
SPORT = 'soccer_epl'

#Looking at EU and UK. May change to just UK moving forward 
REGIONS = 'uk,eu' # uk | us | eu | au. Multiple can be specified if comma delimited

#Only interested in H2H odds between the two teams. May look at how these odds change over time
MARKETS = 'h2h' # h2h | spreads | totals. Multiple can be specified if comma delimited

ODDS_FORMAT = 'decimal' # decimal | american

DATE_FORMAT = 'iso' # iso | unix


#Accessing the API
sports_response = requests.get(
    'https://api.the-odds-api.com/v4/sports', 
    params={
        'api_key': API_KEY
    }
)

#Checking to see what response I get back from the API. Essentially whether I have successfully connected to the API or not
if sports_response.status_code != 200:
    print(f'Failed to get sports: status_code {sports_response.status_code}, response body {sports_response.text}')

else:
    print('List of in season sports:', sports_response.json())



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
# Now get a list of live & upcoming games for the sport you want, along with odds for different bookmakers
# This will deduct from the usage quota
# The usage quota cost = [number of markets specified] x [number of regions specified]
# For examples of usage quota costs, see https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

odds_response = requests.get(
    f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds',
    params={
        'api_key': API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': DATE_FORMAT,
    }
)

if odds_response.status_code != 200:
    print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

else:
    odds_json = odds_response.json()
    print('Number of events:', len(odds_json))
    print(odds_json)

    # Check the usage quota
    print('Remaining requests', odds_response.headers['x-requests-remaining'])
    print('Used requests', odds_response.headers['x-requests-used'])

#Instantiating a list to store the odds dataframes for each of the games
match_odds_list = []
for game in odds_json:  
    #For testing purposes 
    #game = odds_json[0]
    
    #Extracting the game data from dictionary for each of the games
    home_team = game['home_team']
    away_team = game['away_team']
    kick_off_time = game['commence_time']
    bookmakers = game['bookmakers']
    
    #Extracting the odds from each of the bookmakers
    
    #Instantiating bookmaker odds list
    bookmaker_odds_list = []
    
    for bookmaker in bookmakers:
        #bookmaker = bookmakers[0]
        
        #Extracting the bookmaker name
        bookmaker_name = bookmaker['title']
        odds = bookmaker['markets'][0]['outcomes']
        for odd in odds:
            if odd['name'] == home_team:
                home_team_odds = odd['price']
            elif odd['name'] == away_team:
                away_team_odds = odd['price']
            elif odd['name'] == 'Draw':
                draw_odds = odd['price']
            else:
                print("Error")
                
        #Creating a dictionary to store the odds for the game that will then be joined into a dataframe
        odds_dict = {"home_team": home_team, "away_team": away_team, "date": kick_off_time, "bookmaker": bookmaker_name, "odds_h": home_team_odds, "odds_a": away_team_odds, "odds_d": draw_odds}        
        odds_df = pd.DataFrame(odds_dict, index=[0])

        #Renaming the columns to include the bookmaker name
        odds_df = odds_df.rename(columns={"odds_h": f"odds_h_{bookmaker_name}", "odds_a": f"odds_a_{bookmaker_name}", "odds_d": f"odds_d_{bookmaker_name}"})

        #dropping bookmaker column
        odds_df = odds_df.drop(columns=["bookmaker"])
        
        #Appending the odds_df to the bookmaker_odds_list
        bookmaker_odds_list.append(odds_df)
        
    #Merging all the odds dataframes together by home team, away team, and kick off time
    all_bookmaker_odds = reduce(lambda left,right: pd.merge(left, right, on=["home_team", "away_team", "date"]), bookmaker_odds_list)
    
    match_odds_list.append(all_bookmaker_odds)  
    

#Concatenating all the match odds dataframes together
all_match_odds = pd.concat(match_odds_list, ignore_index=True, axis=0, sort=False)

#Reducing the dataset down to only include pinnacle and william hill odds as that is all I have in my historical dataset
all_match_odds = all_match_odds[["home_team", "away_team", "date", 
                                 #"odds_h_Pinnacle Sports", "odds_a_Pinnacle Sports", "odds_d_Pinnacle Sports", 
                                 "odds_h_William Hill", "odds_a_William Hill", "odds_d_William Hill"]]


#Creating a column to store the average odds for each game
all_match_odds["odds_h_avg"] = all_match_odds[["odds_h_Pinnacle Sports", "odds_h_William Hill"]].mean(axis=1)
all_match_odds["odds_a_avg"] = all_match_odds[["odds_a_Pinnacle Sports", "odds_a_William Hill"]].mean(axis=1)
all_match_odds["odds_d_avg"] = all_match_odds[["odds_d_Pinnacle Sports", "odds_d_William Hill"]].mean(axis=1)

#Converting the date column to a datetime object
all_match_odds["date"] = pd.to_datetime(all_match_odds["date"], format="%d/%m/%Y")

#Dropping the william hill odds columns as I only wanted it for the average
all_match_odds = all_match_odds.drop(columns=["odds_h_William Hill", "odds_a_William Hill", "odds_d_William Hill"])
#Renaming some of the columns to match the naming convention of the historical dataset
all_match_odds = all_match_odds.rename(columns={"odds_h_Pinnacle Sports": "odds_h_pinnacle", "odds_a_Pinnacle Sports": "odds_a_pinnacle", "odds_d_Pinnacle Sports": "odds_d_pinnacle"})

#Renaming the team names to the full team names that all the other datasets use
#Renaming the team names to the full team names that all the other datasets use
team_name_lookup = pd.read_excel(f"{raw}/team_name_lookup.xlsx")

#Merging the team name lookup table with the betting data
all_match_odds = pd.merge(all_match_odds, team_name_lookup, left_on="home_team", right_on="alternate_name", how="left")
all_match_odds = all_match_odds.drop(columns=["alternate_name"]).rename(columns={"correct_name": "home_team_full_name"})

all_match_odds = pd.merge(all_match_odds, team_name_lookup, left_on="away_team", right_on="alternate_name", how="left")
all_match_odds = all_match_odds.drop(columns=["alternate_name"]).rename(columns={"correct_name": "away_team_full_name"})

#Dropping the home_team and away_team columns
all_match_odds = all_match_odds.drop(columns=["home_team", "away_team"])

#Reordering the columns
all_match_odds = all_match_odds[["date", "home_team_full_name", "away_team_full_name", 
                                                           "odds_h_avg", "odds_a_avg", "odds_d_avg", 
                                                           "odds_h_pinnacle", "odds_a_pinnacle", "odds_d_pinnacle"]]

#Writing to a feather file in the intermediate folder
all_match_odds.to_feather(f"{intermediate}/all_match_odds.feather")

