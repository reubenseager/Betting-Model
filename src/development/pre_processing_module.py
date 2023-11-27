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
from functools import reduce
from datetime import datetime
import pyarrow.feather as feather   # Package to store dataframes in a binary format

os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")


#Project directory locations
raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"

elo_data_folder = Path(intermediate, "elo_data")
webscraped_football_data_folder = Path(intermediate, "webscraped_football_data")
historical_betting_odds = Path(intermediate, "historical_betting_odds")
live_betting_odds_folder = Path(intermediate, "live_betting_odds")



#TODO = Rolling team average XG (Expected Goals)
#TODO = Do I wanna keep the time of match variable. (Think probably not)
#TODO = Need to keep a time from last game variable 
#TODO = squad value (https://www.footballtransfers.com/en/values/teams/most-valuable-teams)
#TODO = Figure out if cat.codes gives the same codes to the same teams if we recreate the model next week. Maybe look ayt using sklearn encoders
#TODO = Filter to occasions where both the home and away prediction (of the same game) are the same. 
#TODO = average percentage of stadium capacity filled
#TODO = Sentiment analysis of twitter data (https://www.scraperapi.com/resources/)
#TODO = Shots conceded last 5 games
#TODO = Think about getting rid of the team and opponent code. Not entirely comfortable with having the teams as labels in gthe model. Maybe look with and without it.
#TODO = Look into using pipeline
#TODO = Look into having the position of the team in the league as a feature. Maybe add more weight to this feature as the season progresses. So later on in the season theres more weight.
#TODO = Try ohe the opponent and tyeam in one of the trials. Probably with pipeline to see if it benefits the model.
#TODO = Once the pipeline is writte, I should probably maybe run it seperately for test and train. Althought this would reduce my data. Also the data preprocessing work on previous data rather than all data
#TODO = Try different input datastes. FOr example, try with and wihtout ohg, label encoding, removing missing rows in previous club performance, filling them in etc
#TODO = Look at adding in betting predictors from the different sports books as features in the model.
#TODO = Elo weighted passed performance of points. 
#TODO = Do they do well against these types of teams.Group teams on style. Maybe do some clusteirng on teams to find similar
# pipeline stuff: https://www.freecodecamp.org/news/machine-learning-pipeline/


#! = Need to have all results for both the home and away team on a single line I think. So just copy all the information that could be attributed ot both, like recent form, xg for last 5 games, maybe an xg diff for and away for tht last few games for both home and away team (https://medium.com/codex/football-analytics-using-bayesian-network-for-analyzing-xg-expected-goals-705e63e597c2)

#The web scraping and ELO ratings scripts shouldn't need to be run more than once. I should be able to work from this script only.

#Start by loading the match data, elo data, historical betting data, current betting data, and team naming data. These will be joined together using the home team name, away team name and date columns.
elo_ratings_all_teams = pd.read_feather(f"{elo_data_folder}/elo_ratings_all_teams.feather")
match_data_all_teams  = pd.read_feather(f"{webscraped_football_data_folder}/match_data_all_teams.feather")
historical_betting_all_teams = pd.read_feather(f"{historical_betting_odds}/all_historical_betting_data.feather")
live_betting_all_teams = pd.read_feather(f"{live_betting_odds_folder}/all_match_odds.feather")

#Converting the date column to a datetime object
historical_betting_all_teams["date"]= historical_betting_all_teams["date"].apply(lambda x: x.date())
elo_ratings_all_teams["date"]= elo_ratings_all_teams["date"].apply(lambda x: x.date())

team_name_lookup = pd.read_excel(f"{raw}/team_name_lookup.xlsx")

#Joining datasets together into a single combined dataframe (Can add more datasets to this later on)


#Merging the team_name_lookup_df to the match_df dataframe (Should probably move this to the web scraping script eventually)
match_data_all_teams = match_data_all_teams.merge(team_name_lookup, how="left", left_on="team", right_on="alternate_name")
match_data_all_teams = match_data_all_teams.drop(columns=["alternate_name"]).rename(columns={"correct_name": "team_full_name"})

match_data_all_teams = match_data_all_teams.merge(team_name_lookup, how="left", left_on="opponent", right_on="alternate_name")
match_data_all_teams = match_data_all_teams.drop(columns=["alternate_name"]).rename(columns={"correct_name": "opponent_full_name"})

#Removing duplicate rows
match_data_all_teams = match_data_all_teams.drop_duplicates()


#Testing that there are no NaN values for the team names in the data
match_data_all_teams[["team_full_name", "opponent_full_name"]].isnull().sum()

#Drop the team and opponent columns as they are no longer needed
match_data_all_teams = match_data_all_teams.drop(columns=["team", "opponent"])

match_data_all_teams['date'] = match_data_all_teams['date'].apply(lambda x: x.date())

#Creating a unique game id column that includes date, team and opponent
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

#Dropping duplicate rows
elo_ratings_all_teams = elo_ratings_all_teams.drop_duplicates()

#Testing that there are no NaN values in the data
elo_ratings_all_teams.isnull().sum()



#Now merging the elo_df_all_dates dataframe to the match_df dataframe
match_data_all_teams = match_data_all_teams.merge(elo_ratings_all_teams, how = "left", left_on=["team_full_name", "date"], right_on=["elo_team_name", "date"]).rename(columns={"elo_points": "team_elo_points"}).drop(columns=["elo_team_name"])
match_data_all_teams = match_data_all_teams.merge(elo_ratings_all_teams, how = "left", left_on=["opponent_full_name", "date"], right_on=["elo_team_name", "date"]).rename(columns={"elo_points": "opponent_elo_points"}).drop(columns=["elo_team_name"])


#Dropping any duplicate rows
match_data_all_teams = match_data_all_teams.drop_duplicates()


#Dropping the excess name columns
match_data_all_teams = match_data_all_teams.drop(columns=["gls", "comp", "day"], axis=1,   errors="ignore")
#complete_data = complete_data.drop(columns=["match_team_names", "elo_team_names", "club", "gls", "index"], axis=1,   errors="ignore")


#Seeing how many games there are per team using value counts
#match_data_all_teams["team"].value_counts()


#Data Cleaning & Feature Engineering

#SHould probably do some checks for null values in the data. Should also probably do some plots to see if there are any outliers in the data
match_data_all_teams.isnull().sum()

#Do some type of pytest here to check that the data is in the correct format

#If dist is missing and has a nan value, then fill it with the highest value for that team. This is because dist being nan means the team has had no shots. Which should be punished by the model
match_data_all_teams["dist"] = match_data_all_teams.groupby("team_full_name")["dist"].transform(lambda x: x.fillna(x.max()))

#Again checking whether there are any missing values in the data
match_data_all_teams.isnull().sum()

#Writing a test to check that there are no missing values in the data. Print out which columns are missing values if there are any (NOT sure if this works or how you use pytest)
def test_no_missing_values():
    assert match_data_all_teams.isnull().sum().sum() == 0, "There are missing values in the data"
    
test_no_missing_values()
  

#Checking data types. Need numeric data as that is what ML models take as input
match_data_all_teams.dtypes

#Converting venue into a numeric coded variable
# complete_data["venue"] = complete_data["venue"].astype("category")
# complete_data["venue_code"] = complete_data["venue"].cat.codes

#Creating a code for the opponent column
# complete_data["opponent"] = complete_data["opponent"].astype("category")
# complete_data["opponent_code"] = complete_data["opponent"].cat.codes

#Creating a points column where a W is 3 points, a D is 1 point and a L is 0 points
match_data_all_teams["points"] = match_data_all_teams["result"].map({"W":3, "D":1, "L":0})



####################################
#Current league position
####################################
#Possibly look at including some type of gameweek value in there. SO that closer to the end of the season, the model gives more weight to the current league position

#Calculating the cumulative points for each team up to the current game by season
match_data_all_teams["cumulative_points"] = match_data_all_teams.groupby(["team_full_name", "season"])["points"].cumsum()

#Creating a gameweek column. This is the number of games that the team has played in the season. The +1 is there because the first game of the season is gameweek 1 not 0
match_data_all_teams["gameweek"] = match_data_all_teams.groupby(["team_full_name", "season"])["points"].cumcount() + 1

#Creating a current league position column. This the the rank of cumulative points for the specific gameweek and season. If there are two teams with the same number of points, then I'm just giving them the same league position
match_data_all_teams["current_league_position"] = match_data_all_teams.groupby(["season", "gameweek"])["cumulative_points"].rank(ascending=False, method="min", pct=False, na_option="keep") 

#Creating a "weighted" current league position column. This gives more weight to positions that are later on in the season
match_data_all_teams["weighted_league_position"] = ((20 - match_data_all_teams["current_league_position"] + 1) * (1 + match_data_all_teams["gameweek"]/38))

#Drop the cumulative points and gameweek columns as they are no longer needed
match_data_all_teams.drop(columns = ["cumulative_points", "gameweek", "current_league_position"], axis=1, inplace=True)


####################################
#Rolling Averages
####################################

#Function that will create a rolling average of the data for the last 5 games for each team
#TODO = Maybe look at having a minimum winodw of 3 or something here to reduce the number of missing datapoints
def rolling_averages(group, cols, new_cols, window):
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
match_data_all_teams = match_data_all_teams.groupby("team_full_name").apply(lambda x: rolling_averages(x, cols=cols_for_rolling, new_cols=new_rolling_cols, window=5))

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

match_data_all_teams = match_data_all_teams.groupby(["team_full_name", "venue"]).apply(lambda x: rolling_averages(x, cols=["points"], new_cols=["venue_points"], window=3)).droplevel(["team_full_name", "venue"])


####################################
#Previous performance against opponent
####################################
#Here I will be looking at how well the team has done against the previous club over their last 5 max but will allow just previous fixture fixtures.
#Maybe also look at this home and away but only for the most recent game 

match_data_all_teams = match_data_all_teams.groupby(["team_full_name", "opponent_full_name"]).apply(lambda x: rolling_averages(x, cols=["points"], new_cols=["points_against_opponent"], window=2)).droplevel(["team_full_name", "opponent_full_name"])

####################################
#Average ELO of past 5 opponents
####################################
#This should help give some context to the team's recent form. As if they have been playing against teams with a high elo rating, then their points are likely to be lower.

match_data_all_teams = match_data_all_teams.groupby("team_full_name").apply(lambda x: rolling_averages(x, cols=["opponent_elo_points"], new_cols=["average_opponent_elo"], window=5)).droplevel("team_full_name")

####################################
#Betting odds data
####################################
#Need to combine the historical data with the live data

#Reading in the betting data
all_betting_data = feather.read_feather(f"{intermediate}/all_betting_data.feather")

#Updating the betting data with the latest betting odds
all_betting_data = pd.concat([all_betting_data, live_betting_all_teams], axis=0, ignore_index=True, join="inner").drop_duplicates(subset=['date', 'home_team_full_name', 'away_team_full_name'],keep='last')

feather.write_feather(df=all_betting_data, dest=f"{intermediate}/all_betting_data.feather")

#Removing the hour and minute from the date column but keep UTC timezone
all_betting_data['date'] = all_betting_data['date'].apply(lambda x: x.date())

all_betting_data["game_id"] = all_betting_data.apply(
    lambda row: (
        str(row["date"]) + "_" + row["home_team_full_name"] + "_" + row["away_team_full_name"]), axis=1
)



match_data_all_teams = match_data_all_teams.merge(all_betting_data[[col for col in all_betting_data.columns if col != "date"]], 
                                  how = "left", 
                                  left_on=["team_full_name", "opponent_full_name" ,"game_id"], 
                                  right_on=["home_team_full_name", "away_team_full_name" ,"game_id"]).drop(columns=["home_team_full_name", "away_team_full_name"])


#Reducing the all_football_data dataframe to only include the columns that we want to use in the model
# cols_for_model = ["date", "elo_rank", "elo_points", "venue_code", "opponent_code", "gf_rolling", "ga_rolling" ,"xg_rolling", "xga_rolling",
#                  "npxg_rolling", "points_rolling", "sh_rolling", "poss_rolling", "sot_rolling", "dist_rolling", "home_away_ppg_rolling", 
#                  "rolling_previous_performance", "rolling_elo_opponents", "result"]

# match_data_all_teams = match_data_all_teams[cols_for_model]


#The rolling_previous_performance column has some null values. This is because the team has not played the opponent before. So I will fill these with the corresponding value from points_rolling column
#Not actually sure if this is somethign I should do. As its adding information to the model that doesn't exist in real life but may be dropping too many rows if I don't do this.
#all_football_data["rolling_previous_performance"] = all_football_data["rolling_previous_performance"].fillna(all_football_data["points_rolling"])
#all_football_data["home_away_ppg_rolling"] = all_football_data["home_away_ppg_rolling"].fillna(all_football_data["points_rolling"])


#Creating a dataset that contains the home and away data for a match on the same row

#These are the columns that will be used for the home and away teams
home_and_away_cols = ["team_full_name", "gf_rolling", "ga_rolling", "xg_rolling", "xga_rolling", "npxg_rolling", "points_rolling", "sh_rolling",
                      "poss_rolling", "sot_rolling", "dist_rolling", "team_elo_points" ,"points_against_opponent", "average_opponent_elo", "venue_points", "weighted_league_position"]

#These are the betting columns. They include information for both the home and away teams
betting_cols = [col for col in match_data_all_teams.columns if col.startswith("odds_")]


#Creating a dataframe that contains the home team data (Also including the results column)
home_team_data = match_data_all_teams[match_data_all_teams["venue"] == "Home"][["date", "game_id", "result"] + home_and_away_cols]

#renaming the columns to include the home suffix
home_team_data = home_team_data.rename(columns={col: f"{col}_home_team" for col in home_and_away_cols})

#Doing the same for the away team data. But not including the result column as this i in the home team data
away_team_data = match_data_all_teams[match_data_all_teams["venue"] == "Away"][["date", "game_id"] + home_and_away_cols]

away_team_data = away_team_data.rename(columns={col: f"{col}_away_team" for col in home_and_away_cols})

betting_data = match_data_all_teams[match_data_all_teams["venue"] == "Home"][["date", "game_id"] + betting_cols]


#Merging all the data together using reduce
all_football_data = reduce(lambda left, right: pd.merge(left, right, on=["date", "game_id"]), [home_team_data, away_team_data, betting_data])


#Checking the data types of the all_football_data dataframe
all_football_data.dtypes

#Converting the result column to a categorical variable
all_football_data["result"] = all_football_data["result"].astype("category")

#Checking the data types of the all_football_data dataframe
all_football_data.dtypes

#Chcking for missing values in the all_football_data dataframe. There are always going to be some missing values as I am taking rolling averages.
all_football_data.isnull().sum()

#Removing all rows where points_agains_opponent is null. This is because the team has not played the opponent before. So I will fill these with the corresponding value from points_rolling column  
all_football_data = all_football_data.dropna(subset=["points_against_opponent_home_team"])

all_football_data.isnull().sum()

#Dropping the columns that have missing values. As this will cause issues for the ML models.
all_football_data = all_football_data.dropna()

#Chcking for missing values in the all_football_data dataframe. There are always going to be some missing values as I am taking rolling averages.
all_football_data.isnull().sum()

#Columns to keep
cols_for_model = ["date", "result"] + [col for col in all_football_data.columns if col.startswith(tuple(home_and_away_cols)) and not col.startswith("team_full_name")] + [col for col in all_football_data.columns if col in betting_cols]


all_football_data = all_football_data[cols_for_model]

#Writing the all_football_data dataframe to a feather file. This is essentially my input data for the model
feather.write_feather(df=all_football_data, dest=f"{intermediate}/all_football_data.feather")



####################################
#Feature Selection
####################################
#TODO = Look at using things like ANOVA to select the best features for the model.
#TODO = Look at using Genetic algorithms for feature selection (https://towardsdatascience.com/feature-selection-with-genetic-algorithms-7dd7e02dd237#:~:text=Genetic%20algorithms%20use%20an%20approach,model%20for%20the%20target%20task.)

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
