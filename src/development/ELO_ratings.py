"""
This script is looking at getting ELO data from clubelo.com. This is a website that ranks football teams based on their performance.
I will be using an API (Application Programming Interface) to get the data from the website. This is a way of getting data from a website without having to scrape it.
    
"""
import io # This is a way of reading and writing strings to a buffer (memory) rather than a file
import requests # This is a way of requesting data from a website
import pandas as pd # This is a package for data manipulation and analysis

#Requesting the data from the API

#ELO ranking

#Testing for a single team
test_team = "Man City"
elo_request = requests.get("http://api.clubelo.com/ManCity")

#Ideally want to see a 200 response code which means that the request was successful
print(elo_request.status_code)

#Looking at what the data looks like
print(elo_request.text)

elo_data = io.StringIO(elo_request.text)

elo_df = pd.read_csv(elo_data, sep=",")

elo_df.tail()

#Checking the data types of the elo_df dataframe
elo_df.dtypes

#Converting the "From" and "To" date column to a datetime objects
elo_df["From"] = pd.to_datetime(elo_df["From"])
elo_df["To"] = pd.to_datetime(elo_df["To"])

#Converting the datetimes to YYYY-MM-DD format
elo_df["From"] = elo_df["From"].dt.strftime("%Y-%m-%d")
elo_df["To"] = elo_df["To"].dt.strftime("%Y-%m-%d")

#Filtering the data down to only include 2018 onwards. This should be enough data to train the model on
elo_df = elo_df[elo_df["From"] >= "2018-01-01"]


#Checking the data types of the elo_df dataframe
elo_df.dtypes

#Converting the columns to lowercase
elo_df.columns = [c.lower() for c in elo_df.columns]

#Dropping unwanted columns
elo_df = elo_df.drop(columns=["country", "level", "to"])

#Renaming some of the columns to more user friendly names
elo_df = elo_df.rename(columns={"from": "date", "rank": "elo_rank", "elo": "elo_points"})


#Creating a dataframe with every possible dat between 2018-01-01 and the latest elo date
all_dates = pd.date_range(start=elo_df["date"].min(), end=elo_df["date"].max(), freq="D")

#Converting the all_dates series to a dataframe and naming the column "date"
all_dates = pd.DataFrame(all_dates, columns=["date"])
all_dates["date"] = all_dates["date"].dt.strftime("%Y-%m-%d")

#Joining the all_dates dataframe to the elo_df dataframe
elo_df_all_dates = all_dates.merge(elo_df, how="left", on="date")

#Interpolate the elo_rank and elo_points columns. Think it's okay to have a linear interpolation here as the data is a time series
elo_df_all_dates["elo_rank"] = elo_df_all_dates["elo_rank"].interpolate(method="linear")
elo_df_all_dates["elo_points"] = elo_df_all_dates["elo_points"].interpolate(method="linear")

#If club is null, then fill it with the specific value of club name. (Need to have this as part of the loop variables)
elo_df_all_dates["club"] = elo_df_all_dates["club"].fillna(value=test_team)




def elo_ratings_function(team_name):
    
    
    #Requesting the data from the API
    elo_request = requests.get(f"http://api.clubelo.com/{team_name}")

    elo_data = io.StringIO(elo_request.text)

    elo_df = pd.read_csv(elo_data, sep=",")

    #Converting the "From" date column to a datetime objects
    elo_df["From"] = pd.to_datetime(elo_df["From"])

    #Converting the datetimes to YYYY-MM-DD format
    elo_df["From"] = elo_df["From"].dt.strftime("%Y-%m-%d")

    #Filtering the data down to only include 2018 onwards. This matches the dates for the match data information
    elo_df = elo_df[elo_df["From"] >= "2018-01-01"]

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
    feather.write_feather(df=elo_df_all_dates, dest=f"{intermediate}/{team_name}_elo_ratings.feather")
    
    time.sleep(15)  # A lot of websites allow scraping but don't want you to do it too quickly, so you don't slow down the website

 
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
for team in teams:
    #If the file already exists, then don't run the function
    if os.path.isfile(f"{intermediate}/{team}_elo_ratings.feather"):
        continue
    else:
        elo_ratings_function(team)

#Names of teams in the match_df dataframe and 
#Find uinique team names in the match_df dataframe and convert to a dataframe with the column name "team"
match_df_team_names = pd.DataFrame(match_df["team"].unique(), columns=["match_team_names"])

elo_ratings_team_names = pd.DataFrame(teams, columns=["elo_team_names"])

#Attempted fuzzy matching to link the match data to the elo data but didn't work. Used a manual lookup instead.
team_name_lookup_dict = {"elo_team_names" : ["ManCity", "Liverpool", "Arsenal", "Newcastle", "ManUnited", 
                                 "Tottenham", "AstonVilla", "Brentford", "Brighton", "WestHam", "Chelsea",
                                 "CrystalPalace", "Fulham", "Wolves", "Burnley", "Everton", "Forest", "Bournemouth",
                                 "SheffieldUnited", "Luton", "Leicester", "Leeds", "Southampton", "Coventry", 
                                 "Norwich", "WestBrom", "Middlesbrough", "Millwall", "Swansea", "Blackburn", 
                                 "Sunderland", "Watford", "Preston", "Hull", "Stoke", "BristolCity", "Huddersfield", 
                                 "Birmingham", "Cardiff", "Ipswich", "Rotherham", "Plymouth", "QPR", "SheffieldWeds"],
                         
                         "match_team_names" : ["Manchester City", "Liverpool", "Arsenal", "Newcastle United", "Manchester United",
                                 "Tottenham Hotspur", "Aston Villa", "Brentford", "Brighton and Hove Albion", "West Ham United", "Chelsea",
                                 "Crystal Palace", "Fulham", "Wolverhampton Wanderers", "Burnley", "Everton", "Nottingham Forest", "Bournemouth",
                                 "Sheffield United", "Luton Town", "Leicester City", "Leeds United", "Southampton", "Coventry City",
                                 "Norwich City", "West Bromwich Albion", "Middlesbrough", "Millwall", "Swansea City", "Blackburn Rovers",
                                 "Sunderland", "Watford", "Preston North End", "Hull City", "Stoke City", "Bristol City", "Huddersfield Town",
                                 "Birmingham City", "Cardiff City", "Ipswich Town", "Rotherham United", "Plymouth Argyle", "Queens Park Rangers", "Sheffield Wednesday"]}

team_name_lookup_df = pd.DataFrame(team_name_lookup_dict)

#Merging the team_name_lookup_df to the match_df dataframe
match_df = match_df.merge(team_name_lookup_df, how="left", left_on="team", right_on="match_team_names")

#Now merging the elo_df_all_dates dataframe to the match_df dataframe
match_df = match_df.merge(elo_df_all_dates, how = "left", left_on=["elo_team_names", "date"], right_on=["club", "date"])

























#ELO prediction data (MAY NOT USE AS DON"T HAVE BACK DATA. COULD START SAVING IT NOW THOUGH AND THEN USE IT IN THE FUTURE)

# elo_pred_request = requests.get("http://api.clubelo.com/Fixtures")

# #Ideally want to see a 200 response code which means that the request was successful
# print(elo_pred_request.status_code)

# #Looking at what the data looks like
# print(elo_pred_request.text)

# elo_pred_data = io.StringIO(elo_pred_request.text)

# elo_pred_df = pd.read_csv(elo_pred_data, sep=",")

# elo_df.tail()
