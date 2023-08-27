#Project Set-Up
"""
Welcome to my Football Predicting/Betting ML Model. This is primarily a thought experiment to improve my ML skills rather
than a serious attempt to make money from betting.

In this project I am attempting to build a machine learning model that accurately predicts
EPL results (Win, Draw, Loss). I will be using data from the 2016/17 season to the 2020/21 season.

Types of models to look at:
    So, I need a model that can pick out non-linear tendencies. Some possible choices include:
        Random forest (Decision tree ensemble method)
        XGBoost (Gradient boosting ensemble method)
        Neural Network (Probably overkill)  
        SVM (Support Vector Machine)
        Ensemble Method (Combining multiple models together to generate an output)


The features that I will be looking at are:

    Team Playing
    Opposition
    Home or Away fixture
    Days since last game
    Sentiment Analysis (How people are feeling about the team) Depending on if I can get access to twitter
    Time of Game (Early kick-off, midweek game etc.)
    Formation (maybe not)
    ELO rating (http://clubelo.com/ENG. They have an API that I can use I believe)

    Rolling team average XG (Expected Goals)
    Rolling team average PointsPerGame (5 games)
    Rolling team average GS (Goals scored last 5 games)
    Rolling team average GA (Goals conceded last 5 games)
    Rolling team average ST (Shots taken last 5 games)
    Rolling team average SC (Shots conceded last 5 games)
    Past for against opposing team (Maybe average points over last 5 games or a GS/GA ratio)
    Maybe look at some other features that I would engineer myself:
    Money spent in transfer window?
    Transfer window sentiment?
    Star player playing/missing?

"""
# Setting up the directories for the project
from pathlib import Path
import os

# Getting the current working directory and then changing it to the root of the project.
# If you would like to run this code on your own machine, please change the path to the root of the project
os.getcwd()
os.chdir("/Users/reubenseager/Data Science Projects/2023/Betting Model")

# Creating directories if they do not already exist

Path("data/raw").mkdir(exist_ok=True)
Path("data/intermediate").mkdir(exist_ok=True)
Path("data/output").mkdir(exist_ok=True)

raw = Path.cwd() / "data" / "raw"
intermediate = Path.cwd() / "data" / "intermediate"
output = Path.cwd() / "data" / "output"
