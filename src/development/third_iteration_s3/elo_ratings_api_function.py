"""
This script is looking at getting ELO data from clubelo.com. This is a website that ranks football teams based on their performance.
I will be using an API (Application Programming Interface) to get the data from the website. This is a way of getting data from a website without having to scrape it.
    
"""
#Specific Imports
import requests
import pandas as pd
import time
import io
from tqdm import tqdm #Package to show progress bar
import awswrangler as wr #Package to interact with s3

def elo_ratings_api_function(s3_bucket_name = "epl-prediction-model-data"):
    
    #Setting this to be the location of the S3 bucket
    elo_data_folder = f"s3://{s3_bucket_name}/data/intermediate/elo_data"

    def elo_ratings_function(team_name):
        
        #Test
        #team_name = "ManCity"
        #Requesting the data from the API
        elo_request = requests.get(f"http://api.clubelo.com/{team_name}")

        elo_data = io.StringIO(elo_request.text)

        elo_df = pd.read_csv(elo_data, sep=",")

        #Converting the "From" date column to a datetime objects
        elo_df["From"] = pd.to_datetime(elo_df["From"])

        #Converting the datetimes to YYYY-MM-DD format
        elo_df["From"] = elo_df["From"].dt.strftime("%Y-%m-%d")

        #Converting the columns to lowercase
        elo_df.columns = [c.lower() for c in elo_df.columns]

        #Dropping unwanted columns
        elo_df = elo_df.drop(columns=["country", "level", "to"])

        #Renaming some of the columns to more user friendly names
        elo_df = elo_df.rename(columns={"from": "date", "rank": "elo_rank", "elo": "elo_points"})

        #Creating a dataframe with every possible date between 2018-01-01 and the latest elo date
        all_dates = pd.date_range(start=elo_df["date"].min(), end=elo_df["date"].max(), freq="D")

        #Converting the all_dates series to a dataframe and naming the column "date"
        all_dates = pd.DataFrame(all_dates, columns=["date"])
        all_dates["date"] = all_dates["date"].dt.strftime("%Y-%m-%d")

        #Joining the all_dates dataframe to the elo_df dataframe
        elo_df_all_dates = all_dates.merge(elo_df, how="left", on="date")

        #Interpolate the elo_rank and elo_points columns. Think it's okay to have a linear interpolation here as the data is a time series
        elo_df_all_dates["elo_rank"] = elo_df_all_dates["elo_rank"].interpolate(method="linear")
        elo_df_all_dates["elo_points"] = elo_df_all_dates["elo_points"].interpolate(method="linear")

        #If club is null, then fill it with the specific value of club name.
        elo_df_all_dates["club"] = elo_df_all_dates["club"].fillna(value=team_name)
        
        #Write the dataframe to a feather file
        wr.s3.to_parquet(df=elo_df_all_dates, path=f"{elo_data_folder}/{team_name}_elo_ratings.parquet")
        
        time.sleep(10)  # A lot of websites allow scraping but don't want you to do it too quickly, so you don't slow down the website

    
    #Creating a list of all the teams that I want to access data for. These team names need to be exact for the API to work.
    # This could probably be done by webscraping the clubelo.com website and getting all the team names from there.
    teams = ["ManCity", "Liverpool", "Arsenal", "Newcastle", "ManUnited", 
            "Tottenham", "AstonVilla", "Brentford", "Brighton", "WestHam", "Chelsea",
            "CrystalPalace", "Fulham", "Wolves", "Burnley", "Everton", "Forest", "Bournemouth",
            "SheffieldUnited", "Luton", "Leicester", "Leeds", "Southampton", "Coventry", 
            "Norwich", "WestBrom", "Middlesbrough", "Millwall", "Swansea", "Blackburn", 
            "Sunderland", "Watford", "Preston", "Hull", "Stoke", "BristolCity", "Huddersfield", 
            "Birmingham", "Cardiff", "Ipswich", "Rotherham", "Plymouth", "QPR", "SheffieldWeds"]



    #Applying the elo_ratings_function to all the teams in the teams list
    for team in tqdm(teams, desc="Downloading the ELO data for the teams in our dataset"): #tqdm is a package that shows a progress bar for a for loop
        elo_ratings_function(team)
                                
    #Concatenating all the elo dataframes together into a single dataframe            
    elo_ratings_all_teams = pd.concat([wr.s3.read_parquet(f"{elo_data_folder}/{team}_elo_ratings.parquet") for team in teams], axis=0)

    #Convert the date column to a datetime object in the same format as the match data
    elo_ratings_all_teams["date"] = pd.to_datetime(elo_ratings_all_teams["date"], format="%Y-%m-%d")
    
    #Fixing some inconsistencies in the team names
    elo_ratings_all_teams["club"] = elo_ratings_all_teams["club"].replace({"AstonVilla": "Aston Villa", 
                                                                        "ManCity": "Man City", 
                                                                        "ManUnited": "Man United", 
                                                                        "CrystalPalace": "Crystal Palace", 
                                                                        "SheffieldUnited": "Sheffield United", 
                                                                        "WestBrom": "West Brom", 
                                                                        "WestHam": "West Ham",
                                                                        "SheffieldWeds": "Sheffield Weds",
                                                                        "BristolCity": "Bristol City"})


    #Writing the elo_ratings_all_teams dataframe to a parquet file
    wr.s3.to_parquet(df=elo_ratings_all_teams, path=f"{elo_data_folder}/elo_ratings_all_teams.parquet")

    print("ELO ratings data created successfully!")
