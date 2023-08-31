#WebScraping the football data from the website FBref.com. This is a football statistics website.
# Imports
# Math and data manipulation imports
import os
from pathlib import Path

import pandas as pd  # Package for data manipulation and analysis
import numpy as np # Package for scientific computing
import lxml 
import html5lib
import glob
# Web scraping imports
import requests  # Used to access and download information from websites
from bs4 import BeautifulSoup # Package for working with html and information parsed using requests
import time # Package to slow down the webscraping process

import pyarrow.feather as feather   # Package to store dataframes in a binary format

#Project directory locations
raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"

#TODO = Need to set this up so it re-reads in the most seasons of data and overwrites the previous data. But leaves the previous seasons alone.


# This is the url of the premier league stats page of fbref

# Starting off by getting the data from the 2020/21 to the 2022/23 season
years = list(range(2024, 2020, -1))

# Instantiating the list of dataframes
all_matches = []
# 24  ins into the webscraping tutorial
standings_url = 'https://fbref.com/en/comps/9/Premier-League-Stats'

for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text)
    standings_table = soup.select("table.stats_table")[0]  # Extracting the information relating to the table of data

    links = [link.get("href") for link in standings_table.find_all("a")]  # href attribute specifies the URL of the page the link goes to
    links = [link for link in links if "/squads/" in link]  # This only extracts the squads links rather than player information. Should be 20 long as 20 BPL teams
    team_urls = [f"https://fbref.com{link}" for link in links]

    previous_season = soup.select("a.prev")[0].get("href")  # Getting the previous season so that data from the year season before can be scraped
    standings_url = f"https://fbref.com/{previous_season}"  # Getting the absolute URL to use

    for team_url in team_urls:
        # For testing purposes
        #team_url = team_urls[0]
        

        team_name = team_url.split(sep="/")[-1].replace("-Stats", "").replace("-", " ")  # Getting the team name in a more user-friendly format
        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures", flavor = 'html5lib')[0]

        # Getting the shooting stats info
        soup = BeautifulSoup(data.text)
        links = [link.get("href") for link in soup.find_all("a")]
        links = [link for link in links if link and "/all_comps/shooting/" in link]  # This only extracts the squads links rather than player information. The if l checks whether the l string is none empy i.e truthy which stops the code throwing errors.
        data = requests.get(f"https://fbref.com{links[0]}")  # Converting the urls to absolute urls
        shooting = pd.read_html(data.text, match="Shooting")[0]  # Reading in our shooting stats table
        shooting.columns = shooting.columns.droplevel()  # There is a multi level index here, so we're dropping the top one

        # Joining the shooting data to the match data. Here I'm using a try except as some teams don't have shooting information
        shooting_features = ["Date", "Gls" ,"Sh", "SoT", "Dist", "npxG"]
        try:
            team_data = matches.merge(shooting[shooting_features], how="inner", on="Date")
        except ValueError:  # If there is a ValueError just continue with the loop and don't do anything else
            continue
        
        #Maybe put in here time, since last game to see if that improves the models performance. 
        # But still only predict Premier League games.

        # Filtering only to premier league matches
        team_data = team_data[team_data["Comp"] == "Premier League"]  # Filtering only to Premier League matches
        team_data["Season"] = year  # Adding in season to the data
        team_data["Team"] = team_name  # Adding in team name to the data
        
        #Saving the data to the intermediate library as a feather file (Not sure if this works yet). Myabe use pathlib

        feather.write_feather(df=team_data, dest=f"{intermediate}/{team_name}_{year}.feather")
        
        # Adding this team data dataframe to the list of all the teams
        #all_matches.append(team_data)
        time.sleep(15)  # A lot of websites allow scraping but don't want you to do it too quickly, so you don't slow down the website


#Next step is to concatenate all the dataframes together

# Start List all the files in the intermediate folder
files = os.listdir(intermediate)

#Instantiate the list of dataframes
all_matches = []

#Append the files together into a single dataframe
for file in files:
    df = feather.read_feather(f"{intermediate}/{file}")
    all_matches.append(df)  
    
#Concatenating the elements of the all_matches list together. So joining all the teams match data together
match_df = pd.concat(all_matches, axis=0) #Axis=0 to specify that we're concatenating column-wise

#Converting the column names to lowercase
match_df.columns = [c.lower() for c in match_df.columns]

#Removing the columns that we don't need such as time, attendance, referee, comp, round, captain, match report
match_df = match_df.drop(columns=["time", "attendance", "referee", "comp", "round", "captain", "match report"])

#Writing to a feather file
feather.write_feather(df=match_df, dest=f"{intermediate}/all_matches.feather")

#Outputting as an Excel file and a csv file in the output folder
match_df.to_excel(f"{intermediate}/matches.xlsx")



# Starting off by getting the data from the 2020/21 to the 2022/23 season
years = list(range(2024, 2020, -1))

# 24  ins into the webscraping tutorial
standings_url = 'https://fbref.com/en/comps/9/Premier-League-Stats'

shooting_features = ["Date", "Gls" ,"Sh", "SoT", "Dist", "npxG"]


for year in years:
    
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text)
    standings_table = soup.select("table.stats_table")[0]  # Extracting the information relating to the table of data

    links = [link.get("href") for link in standings_table.find_all("a")]  # href attribute specifies the URL of the page the link goes to
    links = [link for link in links if "/squads/" in link]  # This only extracts the squads links rather than player information. Should be 20 long as 20 BPL teams
    team_urls = [f"https://fbref.com{link}" for link in links]

    previous_season = soup.select("a.prev")[0].get("href")  # Getting the previous season so that data from the year season before can be scraped
    standings_url = f"https://fbref.com/{previous_season}"  # Getting the absolute URL to use

    for team_url in team_urls:
        
        team_name = team_url.split(sep="/")[-1].replace("-Stats", "").replace("-", " ")  # Getting the team name in a more user-friendly format
        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures", flavor = 'html5lib')[0]

        # Getting the shooting stats info
        soup = BeautifulSoup(data.text)
        links = [link.get("href") for link in soup.find_all("a")]
        links = [link for link in links if link and "/all_comps/shooting/" in link]  # This only extracts the squads links rather than player information. The if l checks whether the l string is none empy i.e truthy which stops the code throwing errors.
        data = requests.get(f"https://fbref.com{links[0]}")  # Converting the urls to absolute urls
        shooting = pd.read_html(data.text, match="Shooting")[0]  # Reading in our shooting stats table
        shooting.columns = shooting.columns.droplevel()  # There is a multi level index here, so we're dropping the top one

        #Here I'm using a try except as some teams don't have shooting information
        try:
            team_data = matches.merge(shooting[shooting_features], how="inner", on="Date")
        except ValueError:  # If there is a ValueError just continue with the loop and don't do anything else
            continue
        
        #Maybe put in here time, since last game to see if that improves the models performance. 
        # But still only predict Premier League games.

        # Filtering only to premier league matches
        team_data = team_data[team_data["Comp"] == "Premier League"]  # Filtering only to Premier League matches
        team_data["Season"] = year  # Adding in season to the data
        team_data["Team"] = team_name  # Adding in team name to the data
        
        #Removing the columns that we don't need such as Time, Attendance, Referee, Round, Captain, Match Report, Formation, Notes
        team_data = team_data.drop(columns=["Time", "Attendance", "Referee", "Round", "Captain", "Match Report", "Formation", "Notes"])
        
        #Saving the data to the intermediate library as a feather file
        feather.write_feather(df=team_data, dest=f"{intermediate}/match_data_{team_name}_{year}.feather")
        
        # Adding this team data dataframe to the list of all the teams
        #all_matches.append(team_data)
        time.sleep(10)  # A lot of websites allow scraping but don't want you to do it too quickly, so you don't slow down the website
    
    

#Next I am going to concatenate all the match data files together
# Start List all the files in the intermediate folder that contain the match data
match_files = glob.glob(f"{intermediate}/match_data*.feather")

#Concatenating the elements of the all_matches list together. So joining all the teams match data together
match_data_all_teams = pd.concat([feather.read_feather(file) for file in match_files], axis=0) #Axis=0 to specify that we're concatenating column-wise

#Convert the column names to lowercase
match_data_all_teams.columns = [c.lower() for c in match_data_all_teams.columns]

#converting data types of columns in the match_data_all_teams dataframe
#Converting the date column to a datetime object
match_data_all_teams["date"] = pd.to_datetime(match_data_all_teams["date"])

#Converting gf, ga, sh, sot, poss, gls to integer columns
integer_cols = ["gf", "ga", "sh", "sot", "poss", "gls"]
match_data_all_teams[integer_cols] = match_data_all_teams[integer_cols].astype(int)

#Saving the data to the intermediate library as a feather file
feather.write_feather(df=match_data_all_teams, dest=f"{intermediate}/match_data_all_teams.feather")

#Outputting as an Excel file and a csv file in the output folder
match_data_all_teams.to_excel(f"{intermediate}/match_data_all_teams.xlsx")



















