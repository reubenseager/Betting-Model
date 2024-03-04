"""
This script is used to create the historical betting odds dataset.


"""
#Specific Imports
import pandas as pd
import requests
import io
import awswrangler as wr

def historical_betting_odds_function(last_season = 2324, reread_data = False, s3_bucket_name = "epl-prediction-model-data"):
    

    #Path to the S3 buckets where I am outputting the historical data. I have both raw and intermediate buckets here
    historical_bettings_odds_raw = f"s3://{s3_bucket_name}/data/raw/historical_betting_odds"
    historical_bettings_odds_int = f"s3://{s3_bucket_name}/data/intermediate/historical_betting_odds"

    #These are the years that we are downloading historical betting data for
    #Maybe look at updatting this to automatically update
    
    def generate_numbers(start):
        return [start - i * 101 for i in range((start - 1314) // 101 + 1)]
        
    if reread_data == True:
        betting_years = generate_numbers(last_season)
        
    else:
        betting_years = [last_season]
    
    for year in betting_years:
        #Downloading the latest historical betting data
        football_odds_url = f"https://www.football-data.co.uk/mmz4281/{year}/E0.csv"
        response = requests.get(football_odds_url)
        data = io.StringIO(response.text)
        hist_odds = pd.read_csv(data)
        
        #Saving to the s3 bucket
        wr.s3.to_excel(hist_odds, f"{historical_bettings_odds_raw}/betting_odds_{year}.xlsx")

    #Listing all the historical betting odds in the raw folder
    historical_betting_odds_files = wr.s3.list_objects(f"{historical_bettings_odds_raw}/betting_odds*.xlsx")
    historical_betting_odds_files.sort(reverse=True) #Sorting the files in reverse order so that the most recent file is first in the list


    def create_betting_data(year_odds):

        #Reading in the Excel file 
        betting_data = wr.s3.read_excel(year_odds)
        
        #Filtering to bookmakers that also exist in the betting API
        betting_cols = ["Date", "HomeTeam", "AwayTeam", 
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
        
    #Creating a list of dataframes for each year of betting data
    list_of_dfs = [create_betting_data(year_odds) for year_odds in historical_betting_odds_files]

    #Concatenating the list of dataframes into a single dataframe
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


    #Renaming the team names to the full team names that all the other datasets use
    team_name_lookup = wr.s3.read_excel(f"s3://{s3_bucket_name}/data/raw/team_name_lookup/team_name_lookup.xlsx")

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
    wr.s3.to_parquet(all_historical_betting_data, f"{historical_bettings_odds_int}/all_historical_betting_data.parquet")

    print("Historical betting odds data created successfully!")

                                                                                