"""
This script webscrapes fifa ratings data from fifaindex.com. 
The individual datasets are then saved to s3 and then combined into a single dataset.
    
"""
#Specific Imports
import requests
from bs4 import BeautifulSoup
import pandas as pd
from dateutil.parser import parse
import awswrangler as wr


def webscraping_fifa_ratings_function(reread_data = False, s3_bucket_name = "epl-prediction-model-data"):
    #Setting this to be the location of the S3 bucket
    webscraped_fifa_data = f"s3://{s3_bucket_name}/data/intermediate/webscraped_fifa_data"

    read_all_data = reread_data

    if read_all_data:
        years = list(range(24,11, -1))  # Update with the desired years
    else:
        years = [24]  # Update with the desired years

    def scrape_teams_attributes(year, _start, _end):
        data = []
        
        print(f"Scraping data for FIFA {year}...")

        for update_num in range(_start, _end, -1):
            
            
            base_url = f'https://www.fifaindex.com/teams/fifa{year}_{update_num}/?league=13&order=desc'
            response = requests.get(base_url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                all_a_elements = soup.find_all('a')
            
                #Getting the date info
                specific_a_elements = soup.find_all('a', href="#")[1]
            
                # Extract the text content (date value) from the <a> tag
                date_value = specific_a_elements.text.strip()

                # Find the table containing team attributes
                table = soup.find('table', class_='table-teams')

                if table:
                    # Extract information about teams
                    rows = table.find_all('tr')

                    for row in rows[1:]:  # Skip the header row
                        columns = row.find_all('td')

                        # Ensure there are at least 8 columns
                        if len(columns) >= 8:
                            # Extract relevant information (team name, att, mid, def, ovr)
                            team_name = columns[1].text.strip()
                            att = columns[3].text.strip()
                            mid = columns[4].text.strip()
                            def_ = columns[5].text.strip()
                            ovr = columns[6].text.strip()

                            # Check if it's a Premier League team
                            if "Premier League" or "Barclays PL" in columns[2].text:
                                data.append([team_name, att, mid, def_, ovr, year, date_value])
                else:
                    print(f"Table not found on the page for {base_url}.")

        if not data:
            print(f"No data found for FIFA {year}.")
            return None

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data, columns=['Team Name', 'Att', 'Mid', 'Def', 'Ovr', 'Year', 'Date'])
        
        #Saving to s3  
        wr.s3.to_parquet(df=df, path=f"{webscraped_fifa_data}/fifa_index_{year}.parquet")
        
        return None

    # Example usage:


    # range(600, 589, -1) #24
    # range(589, 557, -1) #23
    # range(557, 486, -1) #22
    # range(486, 419, -1) #21
    # range(419, 353, -1) #20
    # range(353, 278, -1) #19
    # range(278, 173, -1) #18
    # range(173, 73, -1) #17
    # range(73, 18, -1) #16 
    # range(17, 13, -1) #15
    # range(13, 11, -1) #14
    # range(11, 9, -1) #13
    # range(9, 7, -1) #12
    # range(600, 590, -1) #11
    # range(600, 590, -1) #10
    # range(600, 590, -1) #09


    result_dfs = []  # List to store the resulting DataFrames

    for year in years:
        if year == 24:
            #result_df = scrape_teams_attributes(year, 600, 590)
            result_df = scrape_teams_attributes(year, 700, 590)
        elif year == 23:
            result_df = scrape_teams_attributes(year, 589, 557)
        elif year == 22:
            result_df = scrape_teams_attributes(year, 557, 486)
        elif year == 21:
            result_df = scrape_teams_attributes(year, 486, 419)
        elif year == 20:
            result_df = scrape_teams_attributes(year, 419, 353)
        elif year == 19:
            result_df = scrape_teams_attributes(year, 353, 278)
        elif year == 18:
            result_df = scrape_teams_attributes(year, 278, 173)
        elif year == 17:
            result_df = scrape_teams_attributes(year, 173, 73)
        elif year == 16:
            result_df = scrape_teams_attributes(year, 73, 18)
        elif year == 15:
            result_df = scrape_teams_attributes(year, 17, 13)
        elif year == 14:
            result_df = scrape_teams_attributes(year, 13, 11)
        elif year == 13:
            result_df = scrape_teams_attributes(year, 11, 9)
        elif year == 12:
            result_df = scrape_teams_attributes(year, 9, 7)
        else:
            result_df = None
        
        #result_dfs.append(result_df)  # Append the result_df to the list

    #Listing fifa files
    fifa_index_dfs = wr.s3.list_objects(f"{webscraped_fifa_data}/fifa_index_*.parquet")
    combined_fifa_index = pd.concat([wr.s3.read_parquet(f) for f in fifa_index_dfs], ignore_index=True)

    #Converting the Date column to datetime
    combined_fifa_index['date'] = combined_fifa_index['Date'].apply(lambda x: parse(x))

    #Dropping the Date column
    combined_fifa_index.drop(['Date'], axis=1, inplace=True)

    team_list = combined_fifa_index["Team_Name"].unique().tolist()

    country_list = ['Brazil', 'Germany', 'France', 'Spain', 'Argentina', 'Portugal', 'Belgium', 'Italy', 
                    'England', 'Uruguay', 'Poland', 'Croatia', 'Colombia', 'Mexico', 'Netherlands', 'Russia', 
                    'Chile', 'Denmark', 'Serbia', 'Switzerland', 'Sweden', 'Austria', 'Turkey', 'Peru', 
                    'Senegal', 'Morocco', 'Egypt', 'Japan', 'United States', 'Wales', 'Iceland']

    team_list = [x for x in team_list if x not in country_list]

    #Making sure that we don't haev multiple names for the same team
    replace_dict = {"Spurs": "Tottenham Hotspur", 
                    "Manchester Utd": "Manchester United",
                    "West Ham":"West Ham United", 
                    "Wolves":"Wolverhampton Wanderers",  
                    "Nott'm Forest":"Nottingham Forest",  
                    "Brighton":"Brighton & Hove Albion",  
                    "Newcastle Utd":"Newcastle United"}


    team_list = list(set([replace_dict.get(team, team) for team in team_list]))

    #Getting the min and max dates
    min_date = combined_fifa_index["date"].min()
    max_date = combined_fifa_index["date"].max()

    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")

    all_dates = pd.DataFrame(all_dates, columns=["date"])


    fifa_index_dfs_all_dates = []


    for team_name in team_list:
        
        #selecting data for a specific team
        team_df = combined_fifa_index[combined_fifa_index["Team_Name"] == team_name]
        
        #left joining the team_df to the all_dates dataframe
        team_df_all_dates = all_dates.merge(team_df, how="left", on="date")
        
        #filling the miss
        team_df_all_dates["Team_Name"].fillna(method="ffill", inplace=True)
        
        #Converting the columns to numeric so they can be interpolated
        cols_to_convert = ['Att', 'Mid', 'Def', 'Ovr']
        team_df_all_dates[cols_to_convert] = team_df_all_dates[cols_to_convert].astype(float)
        
        #Interpolate the missing data

        team_df_all_dates["Att"] = team_df_all_dates["Att"].interpolate(method="linear")
        team_df_all_dates["Mid"] = team_df_all_dates["Mid"].interpolate(method="linear")
        team_df_all_dates["Def"] = team_df_all_dates["Def"].interpolate(method="linear")
        team_df_all_dates["Ovr"] = team_df_all_dates["Ovr"].interpolate(method="linear")
        
        #Appending the dataframe to the fifa_index_dfs_all_dates list
        fifa_index_dfs_all_dates.append(team_df_all_dates)
        
        
        
    #Concatenating the dataframes in the fifa_index_dfs_all_dates list
    combined_fifa_index_all_dates = pd.concat(fifa_index_dfs_all_dates)

    #dropping the year column
    combined_fifa_index_all_dates.drop(["Year"], axis=1, inplace=True)

    #There shouldn't be any duplicate values as there should be a single value for a team on a specific date
    combined_fifa_index_all_dates.drop_duplicates(inplace=True)

    combined_fifa_index_all_dates.dropna(inplace=True) 

    wr.s3.to_parquet(df=combined_fifa_index_all_dates, path=f"{webscraped_fifa_data}/combined_fifa_index_all_dates.parquet")
    
    return None
