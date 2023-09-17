import requests  # Used to access and download information from websites
import json  # Used to convert json data into python dictionaries
from functools import reduce  # Used to merge multiple dataframes together
import pandas as pd




#TODO= May need to create the inverse of the games so I have the data for both teams in the game

# An api key is emailed to you when you sign up to a plan
# Get a free API key at https://api.the-odds-api.com/
API_KEY = '2884dc7f2d10772d3e86f853ba262963'

SPORT = 'soccer_epl' # use the sport_key from the /sports endpoint below, or use 'upcoming' to see the next 8 games across all sports

REGIONS = 'uk,eu' # uk | us | eu | au. Multiple can be specified if comma delimited

MARKETS = 'h2h' # h2h | spreads | totals. Multiple can be specified if comma delimited

ODDS_FORMAT = 'decimal' # decimal | american

DATE_FORMAT = 'iso' # iso | unix

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
# First get a list of in-season sports
#   The sport 'key' from the response can be used to get odds in the next request
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

sports_response = requests.get(
    'https://api.the-odds-api.com/v4/sports', 
    params={
        'api_key': API_KEY
    }
)


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



#Think I'm going to loop through the list of betting results for the different games and try and extract the odds from them
data = odds_json[0]

home_team = data['home_team']
away_team = data['away_team']
kick_off_time = data['commence_time']
bookmakers = data['bookmakers']

bookmaker = bookmakers[0]
bookmaker_name = bookmaker['title']
odds = bookmaker['markets'][0]['outcomes']
#Find the odds of home team winning the game by searching through the list of odds
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
odds_dict = {"Home Team": home_team, "Away Team": away_team, "Kick Off Time": kick_off_time, "Bookmaker": bookmaker_name, "Odds_H": home_team_odds, "Odds_A": away_team_odds, "Odds_D": draw_odds}        
odds_df = pd.DataFrame(odds_dict, index=[0])   

#Renaming the columns to include the bookmaker name
odds_df = odds_df.rename(columns={"Odds_H": f"Odds_H_{bookmaker_name}", "Odds_A": f"Odds_A_{bookmaker_name}", "Odds_D": f"Odds_D_{bookmaker_name}"})

#dropping bookmaker column
odds_df = odds_df.drop(columns=["Bookmaker"])

match_odds_list = []
for game in odds_json:   
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
        odds_dict = {"home_team": home_team, "away_team": away_team, "ko_time": kick_off_time, "bookmaker": bookmaker_name, "odds_h": home_team_odds, "odds_a": away_team_odds, "odds_d": draw_odds}        
        odds_df = pd.DataFrame(odds_dict, index=[0])

        #Renaming the columns to include the bookmaker name
        odds_df = odds_df.rename(columns={"odds_h": f"odds_h_{bookmaker_name}", "odds_a": f"odds_a_{bookmaker_name}", "odds_d": f"odds_d_{bookmaker_name}"})

        #dropping bookmaker column
        odds_df = odds_df.drop(columns=["bookmaker"])
        
        #Appending the odds_df to the bookmaker_odds_list
        bookmaker_odds_list.append(odds_df)
        
    #Merging all the odds dataframes together by home team, away team, and kick off time
    all_bookmaker_odds = reduce(lambda left,right: pd.merge(left, right, on=["home_team", "away_team", "ko_time"]), bookmaker_odds_list)
    
    match_odds_list.append(all_bookmaker_odds)  
    

#Concatenating all the match odds dataframes together
all_match_odds = pd.concat(match_odds_list, ignore_index=True, axis=0, sort=False)
