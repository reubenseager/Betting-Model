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

team_name_lookup = pd.read_excel(f"{raw}/team_name_lookup.xlsx")


#Joining datasets together into a single combined dataframe (Can add more datasets to this later on)


#Merging the team_name_lookup_df to the match_df dataframe (Should probably move this to the web scraping script eventually)
match_data_all_teams = match_data_all_teams.merge(team_name_lookup, how="left", left_on="team", right_on="alternate_name")
match_data_all_teams = match_data_all_teams.drop(columns=["alternate_name"]).rename(columns={"correct_name": "home_team_full_name"})

match_data_all_teams = match_data_all_teams.merge(team_name_lookup, how="left", left_on="opponent", right_on="alternate_name")
match_data_all_teams = match_data_all_teams.drop(columns=["alternate_name"]).rename(columns={"correct_name": "away_team_full_name"})

#Removing duplicate rows
match_data_all_teams = match_data_all_teams.drop_duplicates()


#Testing that there are no NaN values for the team names in the data
match_data_all_teams[["home_team_full_name", "away_team_full_name"]].isnull().sum()

#Drop the team and opponent columns as they are no longer needed
match_data_all_teams = match_data_all_teams.drop(columns=["team", "opponent"])

#Renaming the team names to the full team names that all the other datasets use in the elo_ratings_all_teams dataframe
elo_ratings_all_teams = elo_ratings_all_teams.merge(team_name_lookup, how="left", left_on="club", right_on="alternate_name")

#dropping elo_rank as it basically provides same info as elo_points
elo_ratings_all_teams = elo_ratings_all_teams.drop(columns=["elo_rank", "club", "alternate_name"]).rename(columns={"correct_name": "elo_team_name"})

#Dropping duplicate rows
elo_ratings_all_teams = elo_ratings_all_teams.drop_duplicates()

#Testing that there are no NaN values in the data
elo_ratings_all_teams.isnull().sum()



#Now merging the elo_df_all_dates dataframe to the match_df dataframe
match_data_all_teams = match_data_all_teams.merge(elo_ratings_all_teams, how = "left", left_on=["home_team_full_name", "date"], right_on=["elo_team_name", "date"]).rename(columns={"elo_points": "home_team_elo_points"}).drop(columns=["elo_team_name"])
match_data_all_teams = match_data_all_teams.merge(elo_ratings_all_teams, how = "left", left_on=["away_team_full_name", "date"], right_on=["elo_team_name", "date"]).rename(columns={"elo_points": "away_team_elo_points"}).drop(columns=["elo_team_name"])


#Dropping any duplicate rows
match_data_all_teams = match_data_all_teams.drop_duplicates()


#Dropping the excess name columns
match_data_all_teams = match_data_all_teams.drop(columns=["gls", "comp", "day", "season"], axis=1,   errors="ignore")
#complete_data = complete_data.drop(columns=["match_team_names", "elo_team_names", "club", "gls", "index"], axis=1,   errors="ignore")


#Seeing how many games there are per team using value counts
#match_data_all_teams["team"].value_counts()


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
    #group = group.dropna() #Might unccoment this and do the dropping of values later on. Just to keep all the datsets the same length
    return group

cols_for_rolling = ["gf", "ga", "xg", "xga", "npxg" ,"points", "sh", "poss", "sot", "dist"]
new_rolling_cols = [f"{col}_rolling" for col in cols_for_rolling]


#Maybe look at having a minimum winodw of 3 or something here to reduce the number of missing datapoints
complete_data_rolling = complete_data.groupby("team").apply(lambda x: rolling_averages(x, cols=cols_for_rolling, new_cols=new_rolling_cols, window=5))

#Dropping the extra index level
complete_data_rolling = complete_data_rolling.droplevel("team")

#Resetting and then re-assigning the index to ensure we have unique values
complete_data_rolling.reset_index(drop=True)

#Reassignign the index values as we want to ensure we have unique values
complete_data_rolling.index = range(len(complete_data_rolling))

#Dropping the non-rolling averaged versions of the rolling columns
complete_data_rolling = complete_data_rolling.drop(columns=cols_for_rolling, axis=1)

#Write code that calculates the rolling averages for each team for the last 5 games for the different venue code types

####################################
#Home/Away Performance
####################################

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

#Going to join all dataframes on date and team columns. So need to drop the venue and points columns
home_away_rolling_ppg.drop(columns=["points", "venue"], axis=1, inplace=True)

#Merging the home away rolling ppg dataframe to the complete_data dataframe
#complete_data_rolling = complete_data_rolling.merge(home_away_rolling_ppg[["home_away_ppg_rolling"]], how="left", left_index=True, right_index=True)

####################################
#Performance against previous club
####################################
#Here I will be looking at how well the team has done against the previous club over their last 5 max but will allow just previous fixture fixtures. I want to keep the date column as I want to be abel to join by it

rolling_previous_performance_against_club = complete_data[["date", "team", "opponent", "points"]]

#Sorting the data by date
rolling_previous_performance_against_club.sort_values(by="date", ascending=True, inplace=True)

rolling_previous_performance_against_club['rolling_previous_performance'] = rolling_previous_performance_against_club.groupby(["team", "opponent"])["points"].transform(lambda x: x.rolling(window=3, min_periods=1, closed="left").mean())

#Dropping the points column as we no longer need it
rolling_previous_performance_against_club = rolling_previous_performance_against_club.drop(columns=["opponent", "points"], axis=1)

####################################
#Average ELO of past 5 opponents
####################################
rolling_average_elo_of_past_5_opponents = complete_data[["date", "team", "opponent" ,"elo_points"]]

opponent_elo_ratings = complete_data[["date", "team", "opponent" ,"elo_points"]]



#Renaming the elo_points column to be opponent_elo_points
opponent_elo_ratings.rename(columns={"elo_points": "opponent_elo_points"}, inplace=True)

#Merging the opponent_elo_ratings dataframe to the rolling_average_elo_of_past_5_opponents dataframe
rolling_average_elo_of_past_5_opponents = rolling_average_elo_of_past_5_opponents.merge(opponent_elo_ratings, how="left", left_on=["date", "team", "opponent"], right_on=["date", "team", "opponent"])

#Sorting by the date as I'm calculating a rolling average
rolling_average_elo_of_past_5_opponents.sort_values(by="date", ascending=True, inplace=True)

rolling_average_elo_of_past_5_opponents["rolling_elo_opponents"] = rolling_average_elo_of_past_5_opponents.groupby("team")["opponent_elo_points"].transform(lambda x: x.rolling(window=5, min_periods=1, closed="left").mean())  

#Dropping unwanted columns
rolling_average_elo_of_past_5_opponents.drop(columns=["opponent", "elo_points", "opponent_elo_points",], axis=1, inplace=True)

#Merging all the datasets into a single combined dataset
list_of_dfs = [complete_data_rolling, home_away_rolling_ppg, rolling_previous_performance_against_club, rolling_average_elo_of_past_5_opponents]
#Left joining each of the dataframes in the extra dataframes in the list to the complete_data_rolling dataframe and saving to new dataframe called all_football_data
    
all_football_data = reduce(lambda left,right: pd.merge(left, right, on=["date", "team"]), list_of_dfs)

#Reducing the all_football_data dataframe to only include the columns that we want to use in the model
cols_for_model = ["date", "elo_rank", "elo_points", "venue_code", "opponent_code", "gf_rolling", "ga_rolling" ,"xg_rolling", "xga_rolling",
                 "npxg_rolling", "points_rolling", "sh_rolling", "poss_rolling", "sot_rolling", "dist_rolling", "home_away_ppg_rolling", 
                 "rolling_previous_performance", "rolling_elo_opponents", "result"]

all_football_data = all_football_data[cols_for_model]


#The rolling_previous_performance column has some null values. This is because the team has not played the opponent before. So I will fill these with the corresponding value from points_rolling column
all_football_data["rolling_previous_performance"] = all_football_data["rolling_previous_performance"].fillna(all_football_data["points_rolling"])
all_football_data["home_away_ppg_rolling"] = all_football_data["home_away_ppg_rolling"].fillna(all_football_data["points_rolling"])


#Checking the data types of the all_football_data dataframe
all_football_data.dtypes

#Converting the result column to a categorical variable
all_football_data["result"] = all_football_data["result"].astype("category")

#Checking the data types of the all_football_data dataframe
all_football_data.dtypes

#Chcking for missing values in the all_football_data dataframe. There are always going to be some missing values as I am taking rolling averages.
all_football_data.isnull().sum()


#Dropping the columns that have missing values. As this will cause issues for the ML models.
all_football_data = all_football_data.dropna()

#Chcking for missing values in the all_football_data dataframe. There are always going to be some missing values as I am taking rolling averages.
all_football_data.isnull().sum()

print(len(all_football_data))

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
