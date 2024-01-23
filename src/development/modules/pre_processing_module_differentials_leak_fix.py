"""
    The primary purpose of this script is to prepare the data and create features that the model will use to predict EPL results.
    
    The data outputted form this script will then be passed to the feature selection script, which will determine which features to use in the model. (This won't be used in production though)

    #TODO = squad value (https://www.footballtransfers.com/en/values/teams/most-valuable-teams)
    #TODO = average percentage of stadium capacity filled
    #TODO = Sentiment analysis of twitter data (https://www.scraperapi.com/resources/)
    #TODO = Try ohe the opponent and tyeam in one of the trials. Probably with pipeline to see if it benefits the model.
    #TODO = Elo weighted past performance of points. 
    #TODO = Do they do well against these types of teams.Group teams on style. Maybe do some clusteirng on teams to find similar
    #TODO = Maybe look at how many plyers are traded in and out of the team from fpl. (proxy for ease of games and if players are missing)

    This paper says they get 63% accuracy. They use NLP data from match previews on the guardian https://www.southampton.ac.uk/~sem03/IAAI_2020.pdf
    
    #Features that other people have found useful:
    
    Baboota and Kaur (2019) - https://arno.uvt.nl/show.cgi?fid=160932
    
    They have found that in general differential features perform better than home and away features. This is because they have better univariate distribution and reduced dimensions
    Their top features include:
        - Form Diff (This is a custome function that they have created. It is essentially a short term form metric)
        - Home Form
        - Away Form
        - STKPP (Shot on target k games past performance)
        - GD Diff (Goal difference differential) (Done)
        - RelMidField (This si relative midfield strength. This was webscraped from www.fifaindex.com)
        - CKPP (Corner kick k games past performance)
        - GKPP (Goal kick k games past performance)
        - ATGD (Away team goal difference) (Done)
        - HTGD (Home team goal difference) (Done)
        

    #TODO = Features that other people have found useful:
    #Fifa ratings (Dunno how I would get this data but seems to be important in other models)(https://arno.uvt.nl/show.cgi?fid=160932)
    #Difference in league position
    #Home team goal difference (Done)
    #Away team goal difference (Done)
    #Difference in points 
    #Home team points
    #Away Team Points
    #percentage of possible points
    #formation consistency
    #promotion status
    #percent of possible points earned moving average (Done)
    #It appears that differential features may be superior to features for home and away. Thisn is because they have better univariate distribution and reduced dimensions (https://arno.uvt.nl/show.cgi?fid=160932) (baboota and kkaur pdf)
    #Maybe look into doing all differential columns where I can
    #TODO = I need to create this form differential that baboota and kaur use (Page 6). If A wins Form(AJ) = Form(Aj-1) + Y(Bj-1)
    #TODO = Ideally I would have the whole thing in a pipeline so I can see which set up (features, models etc) works best.

#TODO: The web scraping and ELO ratings scripts shouldn't need to be run more than once. I should be able to work from this script only.
#TODO = Look at how using a window of 3 or 5 affects the model.(Ideally 3 would be better as it would mean less missing values)

#TODO: I think I have some data leakage. I think some of the things like gd and weighted league position are using data from the game thats being played.
#TODO: Do I want to carry data over from one season to the next. I think I probably shouldn't. AT the momement the points_perc rolling is getting rid of those values but it's gettign rid of quite alot.
#Think I need to shift them to the values of the previous game. So the current game will be the previous game and the previous game will be the game before that.


#TODO: Look at reducing the window sizes so that I can increase the amount of data that I have
#Use some sort of FPL data. Maybe look at the number of players that are traded in and out of the team.
#This can be a good proxy for the following:
#   - If a player is injured
#   - If a player is suspended
#   - If the team have a difficult/easy game coming up

https://github.com/vaastav/Fantasy-Premier-League/blob/master/data/2022-23/gws/gw10.csv




https://penaltyblog.readthedocs.io/en/latest/models/dixon_coles.html

https://pena.lt/y/2021/08/25/predicting-football-results-using-bayesian-statistics-with-python-and-pymc3/
"""
# Imports
#Path and directory imports
import os
from pathlib import Path

#Data manipulation imports
import pandas as pd  # Package for data manipulation and analysis
import numpy as np # Package for scientific computing
from functools import reduce
from datetime import datetime

#Testing imports
import pytest # Package for testing code

#Data storage imports
import pyarrow.feather as feather   # Package to store dataframes in a binary format
import joblib #Package to save python objects as binary files
#Setting the working directory
os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")


####################################
#Directory and file locations
####################################

#Project directory locations
raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"

#File locations
elo_data_folder = Path(intermediate, "elo_data")
webscraped_football_data_folder = Path(intermediate, "webscraped_football_data")
historical_betting_odds = Path(intermediate, "historical_betting_odds")
live_betting_odds_folder = Path(intermediate, "live_betting_odds")

#Window Size

window_size = 3
#window_size = 5


####################################
#Loading Data
####################################
#Here we are loading the different datasets that have been created in the previous scripts. These will then be joined together to create the final dataset that will be used in the model.
#This includes:
#   - Match data (Includes xg, xga, npxg, npxga, goals, etc)
#   - Elo ratings (EL0 ratings are a globak ratings system for football teams. It is essentially a long term form metric)
#   - Historical betting odds (Includes betting odds from 2014 to 2020)
#   - Live betting odds (Includes betting odds from 2020 to present)
#   - Team name lookup (This is a lookup table taht aims to give consistency to the team names. As the team names are not consistent across the different datasets)

match_data_all_teams  = pd.read_feather(f"{webscraped_football_data_folder}/match_data_all_teams.feather")
elo_ratings_all_teams = pd.read_feather(f"{elo_data_folder}/elo_ratings_all_teams.feather")
historical_betting_all_teams = pd.read_feather(f"{historical_betting_odds}/all_historical_betting_data.feather")
live_betting_all_teams = pd.read_feather(f"{live_betting_odds_folder}/all_match_odds.feather")
team_name_lookup = pd.read_excel(f"{raw}/team_name_lookup.xlsx").drop_duplicates()

#Converting the different date columns to datetime format
match_data_all_teams['date'] = match_data_all_teams['date'].apply(lambda x: x.date())
elo_ratings_all_teams["date"]= elo_ratings_all_teams["date"].apply(lambda x: x.date())
historical_betting_all_teams["date"]= historical_betting_all_teams["date"].apply(lambda x: x.date())
live_betting_all_teams["date"]= live_betting_all_teams["date"].apply(lambda x: x.date())


####################################
#Joining Datasets
####################################
#Here we are joining the different datasets together to create the final dataset that will be used in the model.

#Merging the team_name_lookup_df to the match_df dataframe (Should probably move this to the web scraping script eventually)
match_data_all_teams = match_data_all_teams.merge(team_name_lookup, how="left", left_on="team", right_on="alternate_name")
match_data_all_teams = match_data_all_teams.drop(columns=["alternate_name"]).rename(columns={"correct_name": "team_full_name"})

match_data_all_teams = match_data_all_teams.merge(team_name_lookup, how="left", left_on="opponent", right_on="alternate_name")
match_data_all_teams = match_data_all_teams.drop(columns=["alternate_name"]).rename(columns={"correct_name": "opponent_full_name"})

#Testing that there are no NaN values for the team names in the data
match_data_all_teams[["team", "opponent" ,"team_full_name", "opponent_full_name"]].isnull().sum()

#Dropping the team and opponent columns as they are no longer needed (They have been replaced by the team_full_name and opponent_full_name columns)
match_data_all_teams = match_data_all_teams.drop(columns=["team", "opponent"])


#Creating a unique game id column that includes date, team and opponent (So this will be identical for the two teams in the game)
match_data_all_teams["game_id"] = match_data_all_teams.apply(
    lambda row: (
        str(row["date"]) + "_" + row["team_full_name"] + "_" + row["opponent_full_name"]
        if row["venue"] == "Home"
        else str(row["date"]) + "_" + row["opponent_full_name"] + "_" + row["team_full_name"]
    ),
    axis=1
)


#Renaming the team names to the full team names that all the other datasets use in the elo_ratings_all_teams dataframe
elo_ratings_all_teams = elo_ratings_all_teams.merge(team_name_lookup, how="left", left_on="club", right_on="alternate_name")

#dropping elo_rank as it basically provides same info as elo_points
elo_ratings_all_teams = elo_ratings_all_teams.drop(columns=["elo_rank", "club", "alternate_name"]).rename(columns={"correct_name": "elo_team_name"})

#Dropping duplicate rows (There should be no duplicate rows but just to be safe)
elo_ratings_all_teams = elo_ratings_all_teams.drop_duplicates()

#Testing that there are no NaN values in the data
elo_ratings_all_teams.isnull().sum()



#Now merging the elo_df_all_dates dataframe to the match_df dataframe
match_data_all_teams = match_data_all_teams.merge(elo_ratings_all_teams, how="left", left_on=["team_full_name", "date"], right_on=["elo_team_name", "date"]).rename(columns={"elo_points": "team_elo_points"}).drop(columns=["elo_team_name"]).drop_duplicates()
match_data_all_teams = match_data_all_teams.merge(elo_ratings_all_teams, how="left", left_on=["opponent_full_name", "date"], right_on=["elo_team_name", "date"]).rename(columns={"elo_points": "opponent_elo_points"}).drop(columns=["elo_team_name"]).drop_duplicates()



#Dropping the excess name columns
match_data_all_teams = match_data_all_teams.drop(columns=["gls", "comp", "day"], axis=1,   errors="ignore")
#complete_data = complete_data.drop(columns=["match_team_names", "elo_team_names", "club", "gls", "index"], axis=1,   errors="ignore")

####################################
#DATA CLEANING
####################################
#Here we are cleaning the data and creating features that will be used in the model

#Checking whether there are any missing values in the data
#Should do a proper pytest test here to check that there are no missing values in the data
match_data_all_teams.isnull().sum()

#If dist is missing and has a nan value, then fill it with the highest value for that team. This is because dist being nan means the team has had no shots. Which should be punished by the model
match_data_all_teams["dist"] = match_data_all_teams.groupby("team_full_name")["dist"].transform(lambda x: x.fillna(x.max()))

#Again checking whether there are any missing values in the data
match_data_all_teams.isnull().sum().sum()

if match_data_all_teams.isnull().sum().sum() == 0:
    print("There are no missing values in the data")
else:
    print("There are missing values in the data")
    
#Writing a test to check that there are no missing values in the data. Print out which columns are missing values if there are any (NOT sure if this works or how you use pytest)
def test_no_missing_values():
    assert match_data_all_teams.isnull().sum().sum() == 0, "There are missing values in the data"


test_no_missing_values()
  

#Checking data types. Need all the data to be numeric for the model
match_data_all_teams.dtypes


####################################
#FEATURE ENGINEERING
####################################

#Creating a points column where a W is 3 points, a D is 1 point and a L is 0 points
match_data_all_teams["points"] = match_data_all_teams["result"].map({"W":3, "D":1, "L":0})

####################################
#Current league position
####################################
#Creating a weighted league position column. This gives more weight to positions that are later on in the season

#Calculating the cumulative points for each team up to the current game by season. But excluding the current game
# match_data_all_teams["cumulative_points"] = match_data_all_teams.groupby(["team_full_name", "season"])["points"].cumsum()

# #Creating a gameweek column. This is the number of games that the team has played in the season. The +1 is there because the first game of the season is gameweek 1 not 0
# match_data_all_teams["gameweek"] = match_data_all_teams.groupby(["team_full_name", "season"])["points"].cumcount() + 1

# #Creating a current league position column. This the the rank of cumulative points for the specific gameweek and season. If there are two teams with the same number of points, then I'm just giving them the same league position
# match_data_all_teams["current_league_position"] = match_data_all_teams.groupby(["season", "gameweek"])["cumulative_points"].rank(ascending=False, method="min", pct=False, na_option="keep") 

# #Creating a "weighted" current league position column. This gives more weight to positions that are later on in the season
# match_data_all_teams["weighted_league_position"] = ((20 - match_data_all_teams["current_league_position"] + 1) * (1 + match_data_all_teams["gameweek"]/38))


# Calculate the cumulative points excluding the current game
match_data_all_teams["cumulative_points"] = match_data_all_teams.groupby(["team_full_name", "season"], sort=False)["points"].apply(lambda x: x.shift().cumsum()).values

# Calculate the gameweek that we are predicting for
match_data_all_teams["gameweek"] = match_data_all_teams.groupby(["team_full_name", "season"])["points"].cumcount() + 1

# Calculate the current league position excluding the current game
match_data_all_teams["current_league_position"] = match_data_all_teams.groupby(["season", "gameweek"])["cumulative_points"].rank(ascending=False, method="min", pct=False, na_option="keep")

# Calculate the weighted league position excluding the current game
match_data_all_teams["weighted_league_position"] = ((20 - match_data_all_teams["current_league_position"] + 1) * (1 + match_data_all_teams["gameweek"]/38))



####################################
#Percentage of possible points
####################################

#Calculating the percentage of possible points gained by the team up to the current game by season. This needs to be shifted by 1 as the current game is included in the cumulative points
#match_data_all_teams["perc_points"] = match_data_all_teams["cumulative_points"] / (match_data_all_teams["gameweek"] * 3)
match_data_all_teams["perc_points"] = match_data_all_teams["cumulative_points"] / ((match_data_all_teams["gameweek"] - 1) * 3)


#Rolling average of percentage of possible points
match_data_all_teams["perc_points_rolling"] = match_data_all_teams.groupby(["team_full_name", "season"], sort=False)["points"].apply(lambda x: x.rolling(window = window_size, closed = "left").mean()/3).values

#Drop the cumulative points and gameweek columns as they are no longer needed
match_data_all_teams.drop(columns = ["gameweek", "current_league_position"], axis=1, inplace=True)

####################################
#Goal Difference
####################################

#Calculating the cumulative points for each team up to the current game by season
match_data_all_teams["cumulative_gf"] = match_data_all_teams.groupby(["team_full_name", "season"], sort=False)["gf"].apply(lambda x: x.shift().cumsum()).values
match_data_all_teams["cumulative_ga"] = match_data_all_teams.groupby(["team_full_name", "season"], sort=False)["ga"].apply(lambda x: x.shift().cumsum()).values


#Creating a goal difference column (This will be used to create a gd differential column between the home and away team)
match_data_all_teams["gd"] = match_data_all_teams["cumulative_gf"] - match_data_all_teams["cumulative_ga"]



####################################
#Rolling Averages
####################################

#Function that will create a rolling average of the data for the last 5 games for each team
def rolling_averages(group, cols, new_cols, window):
    #This should not be carried over seasons. So need to be grouped by season. I think the perc points will get rid of those data points tho
    
    
    #Start by sorting the data by date as we are looking at recent form
    group = group.sort_values(by="date")
    
    #Closed = left means that the window will ignore the result of the current game. As we do not want to include future informaion in the model
    rolling_stats = group[cols].rolling(window = window, closed = "left").mean()
    group[new_cols] = rolling_stats
    #group = group.dropna() #Might unccoment this and do the dropping of values later on. Just to keep all the datsets the same length
    return group

cols_for_rolling = ["gf", "ga", "xg", "xga", "npxg" ,"points", "sh", "poss", "sot", "dist"]
new_rolling_cols = [f"{col}_rolling" for col in cols_for_rolling]


#Maybe look at having a minimum winodw of 3 or something here to reduce the number of missing datapoints
match_data_all_teams = match_data_all_teams.groupby("team_full_name").apply(lambda x: rolling_averages(x, cols=cols_for_rolling, new_cols=new_rolling_cols, window=window_size))

#Dropping the extra index level
match_data_all_teams = match_data_all_teams.droplevel("team_full_name")

#Resetting and then re-assigning the index to ensure we have unique values
match_data_all_teams.reset_index(drop=True)


#Dropping the non-rolling averaged versions of the rolling columns
cols_to_drop = [col for col in cols_for_rolling if col != "points"]
match_data_all_teams = match_data_all_teams.drop(columns=cols_to_drop, axis=1)
####################################
#Home/Away Performance
####################################
#This only creates one column that contains the home/away form based on whatever the value is in the venue column.

match_data_all_teams = match_data_all_teams.groupby(["team_full_name", "venue"]).apply(lambda x: rolling_averages(x, cols=["points"], new_cols=["venue_points"], window=window_size)).droplevel(["team_full_name", "venue"])

#feature_cols = joblib.load(f"{intermediate}/fs_columns.save")

####################################
#Team Form
####################################
#This is where I am going to use the form equation created by baboota and kaur. (This is essentially going to be a more short term and EPL specific version of the elo ratings)

####################################
#Pi-rating
####################################
#https://medium.com/@ML_Soccer_Betting/implementing-the-pi-ratings-in-python-d90da10fb070

####################################
#Average ELO of past 5 opponents
####################################

#This should help give some context to the team's recent form. As if they have been playing against teams with a high elo rating, then their points are likely to be lower.
match_data_all_teams = match_data_all_teams.groupby("team_full_name").apply(lambda x: rolling_averages(x, cols=["opponent_elo_points"], new_cols=["average_opponent_elo"], window=window_size)).droplevel("team_full_name")

####################################
#Team form
####################################


####################################
#Betting odds data
####################################
#Here we are combining the historical data with the live betting data.

#Reading in the betting data
all_betting_data = feather.read_feather(f"{intermediate}/all_betting_data.feather")

#Updating the betting data with any missing historical betting odds and the live betting odds
all_betting_data = pd.concat([all_betting_data, historical_betting_all_teams, live_betting_all_teams], axis=0, ignore_index=True, join="inner").drop_duplicates(subset=['date', 'home_team_full_name', 'away_team_full_name'],keep='last')

feather.write_feather(df=all_betting_data, dest=f"{intermediate}/all_betting_data.feather")

#Removing the hour and minute from the date column but keep UTC timezone
#Creating game_id column to join onto the data (This will only be joined to the home data)
all_betting_data["game_id"] = all_betting_data.apply(
    lambda row: (
        str(row["date"]) + "_" + row["home_team_full_name"] + "_" + row["away_team_full_name"]), axis=1
)

# Calculating the implied probabilities from the betting odds
all_betting_data["ip_home_avg"] = 1 / all_betting_data["odds_h_avg"]
all_betting_data["ip_draw_avg"] = 1 / all_betting_data["odds_d_avg"]
all_betting_data["ip_away_avg"] = 1 / all_betting_data["odds_a_avg"]

# Summing the implied probabilities
all_betting_data["ip_total_avg"] = all_betting_data["ip_home_avg"] + all_betting_data["ip_draw_avg"] + all_betting_data["ip_away_avg"]

# Dividing the implied probabilities by the total to get the implied probability for each outcome
all_betting_data[["ip_home_avg", "ip_draw_avg", "ip_away_avg"]] = all_betting_data[["ip_home_avg", "ip_draw_avg", "ip_away_avg"]].div(all_betting_data["ip_total_avg"], axis=0)

#Repeating same as above but for the pinnacle odds
all_betting_data["ip_home_pinnacle"] = 1 / all_betting_data["odds_h_pinnacle"]
all_betting_data["ip_draw_pinnacle"] = 1 / all_betting_data["odds_d_pinnacle"]
all_betting_data["ip_away_pinnacle"] = 1 / all_betting_data["odds_a_pinnacle"]

# Summing the implied probabilities
all_betting_data["ip_total_pinnacle"] = all_betting_data["ip_home_pinnacle"] + all_betting_data["ip_draw_pinnacle"] + all_betting_data["ip_away_pinnacle"]

# Dividing the implied probabilities by the total to get the implied probability for each outcome
all_betting_data[["ip_home_pinnacle", "ip_draw_pinnacle", "ip_away_pinnacle"]] = all_betting_data[["ip_home_pinnacle", "ip_draw_pinnacle", "ip_away_pinnacle"]].div(all_betting_data["ip_total_pinnacle"], axis=0)

#Dropping unwamted all columns starting with odds_
all_betting_data = all_betting_data.drop(columns=[col for col in all_betting_data.columns if col.startswith("odds_")] + ["ip_total_avg", "ip_total_pinnacle"])

match_data_all_teams = match_data_all_teams.merge(all_betting_data[[col for col in all_betting_data.columns if col != "date"]], 
                                  how = "left", 
                                  left_on=["team_full_name", "opponent_full_name" ,"game_id"], 
                                  right_on=["home_team_full_name", "away_team_full_name" ,"game_id"]).drop(columns=["home_team_full_name", "away_team_full_name"])


####################################
#Home and Away Cols
####################################
#For my final dataset, I need to have home and away data on the same row. 

#Creating a dataset that contains the home and away data for a match on the same row

#These are the columns that will be used for the home and away teams
home_and_away_cols = ["team_full_name", "gf_rolling", "ga_rolling", "xg_rolling", "xga_rolling", "npxg_rolling", "points_rolling", "sh_rolling","perc_points_rolling",
                      "poss_rolling", "sot_rolling", "dist_rolling", "gd" , "team_elo_points", "average_opponent_elo", "weighted_league_position", "cumulative_points"]

#These are the betting columns. They include information for both the home and away teams
#betting_cols = [col for col in match_data_all_teams.columns if col.startswith("odds_")]
betting_cols = [col for col in match_data_all_teams.columns if col.startswith("ip_")]


#Creating a dataframe that contains the home team data (Also including the results column)
home_team_data = match_data_all_teams[match_data_all_teams["venue"] == "Home"][["date", "game_id", "result"] + home_and_away_cols]

#renaming the columns to include the home suffix
home_team_data = home_team_data.rename(columns={col: f"{col}_home_team" for col in home_and_away_cols})

#Doing the same for the away team data. But not including the result column as this is in the home team data already
away_team_data = match_data_all_teams[match_data_all_teams["venue"] == "Away"][["date", "game_id"] + home_and_away_cols]

away_team_data = away_team_data.rename(columns={col: f"{col}_away_team" for col in home_and_away_cols})

betting_data = match_data_all_teams[match_data_all_teams["venue"] == "Home"][["date", "game_id"] + betting_cols]

#Merging all the data together into a single dataframe using the reduce function
all_football_data = reduce(lambda left, right: pd.merge(left, right, on=["date", "game_id"]), [home_team_data, away_team_data, betting_data])

####################################
#Differential Columns
####################################
#Prinitng out the columns that are in the all_football_data dataframe
all_football_data.columns

#Replacing all possible data points with differential columns. This should help to reduce the dimensioanlity of our data and also help with the univariate distribution of the data

#creating the differential columns
differential_cols = [col for col in home_and_away_cols if "team_full_name" not in col]

for col in differential_cols:
    all_football_data[f"{col}_differential"] = all_football_data[f"{col}_home_team"] - all_football_data[f"{col}_away_team"]


#Dropping the non-differential columns. These are the columns that start with home_and_away_cols and end with _home_team or _away_team
home_and_away_cols_to_drop = [col for col in all_football_data.columns if col.startswith(tuple(home_and_away_cols)) and not (col.startswith("team_full_name") or col.endswith("differential"))]

#all_football_data = all_football_data.drop(columns=home_and_away_cols_to_drop)
#going to try not doing any feature selection and just using all the columns


#Checking the data types of the all_football_data dataframe
all_football_data.dtypes

#Converting the result column to a categorical variable
all_football_data["result"] = all_football_data["result"].astype("category")

#Checking the data types of the all_football_data dataframe
all_football_data.dtypes

#Chcking for missing values in the all_football_data dataframe. There are always going to be some missing values as I am taking rolling averages.
all_football_data.isnull().sum()

#Removing all rows where points_agains_opponent is null. This is because the team has not played the opponent before. So I will fill these with the corresponding value from points_rolling column  
all_football_data = all_football_data.dropna()

all_football_data.isnull().sum()

#Dropping the columns that have missing values. As this will cause issues for the ML models.
all_football_data = all_football_data.dropna()

#Chcking for missing values in the all_football_data dataframe. There are always going to be some missing values as I am taking rolling averages.
all_football_data.isnull().sum().sum()

#Columns to keep
cols_for_model = ["date", "result"] + [col for col in all_football_data.columns if col.startswith(tuple(home_and_away_cols)) and not col.startswith("team_full_name")] + [col for col in all_football_data.columns if col in betting_cols]


all_football_data = all_football_data[cols_for_model]

#Writing the all_football_data dataframe to a feather file. This is essentially my input data for the model
#feather.write_feather(df=all_football_data, dest=f"{intermediate}/all_football_data.feather")
#feather.write_feather(df=all_football_data, dest=f"{intermediate}/all_football_data_differential.feather")
#feather.write_feather(df=all_football_data, dest=f"{intermediate}/all_football_data_differential_leak_fix.feather")
feather.write_feather(df=all_football_data, dest=f"{intermediate}/all_football_data_differential_leak_fix.feather")

