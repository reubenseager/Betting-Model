#WebScraping the football data from the website FBref.com. This is a football statistics website.
# Imports
# Math and data manipulation imports
import os
from pathlib import Path

import pandas as pd  # Package for data manipulation and analysis
import glob

# Web scraping imports
import requests  # Used to access and download information from websites
from bs4 import BeautifulSoup # Package for working with html and information parsed using requests
import time # Package to slow down the webscraping process

import pyarrow.feather as feather   # Package to store dataframes in a binary format

from tqdm import tqdm # Package to show progress bar


os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")

#Project directory locations
raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"

Path(intermediate, "webscraped_football_data").mkdir(exist_ok=True)

webscraped_football_data_folder = Path(intermediate, "webscraped_football_data")

#TODO = Need to set this up so it re-reads in the most seasons of data and overwrites the previous data. But leaves the previous seasons alone.

#Year Variables (getting extra data as I don't think I have enough if only looking back three years)
latest_year = 2024
# earlist_year = 2020
earlist_year = 2017 #Theres no npxg at some point back in the data so will probs notread it in


reread_data = False #Change this to True if you want to re-read the data for previous seasons. If you are only updating the most recent season of data then leave as False.

#Re-read all data
if reread_data :
    years = years = list(range(latest_year, earlist_year, -1))
else :
    years = list(range(latest_year, latest_year - 1, -1))

#This is the location of the Premier League standings table on the website
standings_url = 'https://fbref.com/en/comps/9/Premier-League-Stats'

#RThis is a list of shooting features that we want to extract from the website
shooting_features = ["Date", "Gls" ,"Sh", "SoT", "Dist", "npxG"]


for year in years:
    
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text, features="lxml")
    standings_table = soup.select("table.stats_table")[0]  # Extracting the information relating to the table of data

    links = [link.get("href") for link in standings_table.find_all("a")]  # href attribute specifies the URL of the page the link goes to
    links = [link for link in links if "/squads/" in link]  # This only extracts the squads links rather than player information. Should be 20 long as 20 BPL teams
    team_urls = [f"https://fbref.com{link}" for link in links]

    previous_season = soup.select("a.prev")[0].get("href")  # Getting the previous season so that data from the year season before can be scraped
    standings_url = f"https://fbref.com/{previous_season}"  # Getting the absolute URL to use

    for team_url in tqdm(team_urls):
        
        team_name = team_url.split(sep="/")[-1].replace("-Stats", "").replace("-", " ")  # Getting the team name in a more user-friendly format
        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures", flavor = 'html5lib')[0]

        # Getting the shooting stats info
        soup = BeautifulSoup(data.text, features="lxml")
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
        feather.write_feather(df=team_data, dest=f"{webscraped_football_data_folder}/match_data_{team_name}_{year}.feather")
        
        # Adding this team data dataframe to the list of all the teams
        #all_matches.append(team_data)
        time.sleep(10)  # A lot of websites allow scraping but don't want you to do it too quickly, so you don't slow down the website
    
    

#Next I am going to concatenate all the match data files together
# Start List all the files in the intermediate folder that contain the match data. (Ignoring the combined match data file)
match_files = glob.glob(f"{webscraped_football_data_folder}/match_data*.feather")
match_files = [file for file in match_files if "match_data_all_teams.feather" not in file]

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
feather.write_feather(df=match_data_all_teams, dest=f"{webscraped_football_data_folder}/match_data_all_teams.feather")

#Outputting as an Excel file
match_data_all_teams.to_excel(f"{webscraped_football_data_folder}/match_data_all_teams.xlsx")



















