
####################################################################################
#Environment Set-Up
####################################################################################

#Imports
import os
from pathlib import Path

# Getting the current working directory and then changing it to the root of the project.
# If you would like to run this code on your own machine, please change the path to the root of the project
os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")

# Creating the primary directories if they do not already exist
Path("data/raw").mkdir(exist_ok=True)
Path("data/intermediate").mkdir(exist_ok=True)
Path("data/output").mkdir(exist_ok=True)

raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"

####################################################################################
#Generating and importing historical odds
####################################################################################

#Imports
import pandas as pd
import pyarrow.feather as feather   # Package to store dataframes in a binary format
import glob

# Running the historical betting odds function
with open(Path.cwd()/"src"/"development"/"historical_betting_odds_module.py") as f:
    code = compile(f.read(), 'historical_betting_odds_module.py', 'exec')
    exec(code)

####################################################################################
#Live Data (ELO Ratings & Live Betting Odds)
####################################################################################

##ELO Ratings

#Imports
import io

import requests  # Used to access and download information from websites
import time # Package to slow down the webscraping process
from tqdm import tqdm # Package to show progress bar

# Running the historical betting odds function
with open(Path.cwd()/"src"/"development"/"ELO_ratings_module.py") as f:
    code = compile(f.read(), 'ELO_ratings_module.py', 'exec')
    exec(code)
    
##Live Odds Ratings

#Imports
from functools import reduce  # Used to merge multiple dataframes together

import requests  # Used to access and download information from websites
import time # Package to slow down the webscraping process
from tqdm import tqdm # Package to show progress bar
from functools import reduce  # Used to merge multiple dataframes together
from datetime import datetime, timedelta # Package to work with dates
from dateutil import parser # Package to work with dates

import pytz

# Running the historical betting odds function
with open(Path.cwd()/"src"/"development"/"odds_api_module.py") as f:
    code = compile(f.read(), 'odds_api_module.py', 'exec')
    exec(code)
