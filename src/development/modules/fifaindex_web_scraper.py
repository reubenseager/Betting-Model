#Imports

#File management imports
import os
from pathlib import Path
import glob

#Data manipulation imports
import pandas as pd

#Web scraping imports
import requests
from bs4 import BeautifulSoup

#Data storage imports
import pyarrow.feather as feather

#Project directory locations
#Setting the working directory
os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")


intermediate = Path.cwd() / "data" / "intermediate"
webscraped_fifaindex_data = intermediate / "webscraped_fifaindex_data"
webscraped_fifaindex_data.mkdir(exist_ok=True)

read_all_data = False

if read_all_data:
    years = list(range(24,11, -1))  # Update with the desired years
else:
    years = [24]  # Update with the desired years


def scrape_teams_attributes(year, _start, _end):
    data = []
    
    print(f"Scraping data for FIFA {year}...")
    # year = 18
    # _start = 278
    # _end = 590
    # update_num = 2
    

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
    
    #saving data to a permanent location
    feather.write_feather(df=df, dest=f"{webscraped_fifaindex_data}/fifa_index_{year}.feather")
    
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


#List of dataframes to concatenate
fifa_index_dfs = glob.glob(f"{webscraped_fifaindex_data}/fifa_index_*.feather")
combined_fifa_index = pd.concat([pd.read_feather(f) for f in fifa_index_dfs], ignore_index=True)

feather.write_feather(df=combined_fifa_index, dest=f"{webscraped_fifaindex_data}/combined_fifa_index.feather")
