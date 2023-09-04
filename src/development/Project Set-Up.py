#Project Set-Up
"""
Welcome to my Football Predicting/Betting ML Model. This is primarily a thought experiment to improve my ML skills rather
than a serious attempt to make money from betting.

In this project I am attempting to build a machine learning model that accurately predicts
EPL results (Win, Draw, Loss). I will be using data from the 2016/17 season to the 2020/21 season.

Types of models to look at:
    So, I need a model that can pick out non-linear tendencies. Some possible choices include:
        Classification Models:
            Random forest (Decision tree ensemble method)
            XGBoost (Gradient boosting ensemble method)
            Neural Network (Keras)  
            SVM (Support Vector Machine)
            Ensemble Method (Combining multiple models together to generate an output)
            Optuna (Hyperparameter tuning)
            looking into predict_proba() to get a probability distribution for each class
            https://towardsdatascience.com/pythons-predict-proba-doesn-t-actually-predict-probabilities-and-how-to-fix-it-f582c21d63fc
            The above artivle talks about how you need to calibrate the probabilites first to get a better output. from sklearn.calibration import calibration_curve

        
        Regression Models:
        For this I will be predicting the goals scored by each team then using a probability distribution to predict the result.
            Poisson distribution
            negative binomial distribution
            weibull count model (https://blogs.salford.ac.uk/business-school/wp-content/uploads/sites/7/2016/09/paper.pdf). This claims to actually make money from betting.
            


The features that I will be looking at are:

    Team Playing
    Opposition
    Home or Away fixture
    Days since last game
    Sentiment Analysis (How people are feeling about the teamhttps://www.scraperapi.com/resources/) Depending on if I can get access to twitter with spending money
    Time of Game (Early kick-off, midweek game etc.)
    Formation (maybe not)
    ELO rating (http://clubelo.com/ENG. They have an API that I can use I believe)
    ELO rating prediction information from (api.clubelo.com/Fixtures. Not sure if this has data for all games)

    Rolling team average XG (Expected Goals) Done
    Rolling team average XGA (Expected Goals Against) Done
    Rolling team average PPG (5 games) Done
    Rolling team average PPG Away (PPG Goals Away) Done (But combined with PPG Home into a single column)
    Rolling team average PPG Home (Expected Goals Against)
    Rolling team average GS (Goals scored last 5 games) Done
    Rolling team average GA (Goals conceded last 5 games) Done
    Rolling team average ST (Shots taken last 5 games) Done
    Rolling team average SC (Shots conceded last 5 games) (NEED TO DO)
    Past for against opposing team (Average PPG last 5 games against specific opposition) (Done)
    Average ELO of past 5 opponents
    Maybe look at some other features that I would engineer myself:
    Money spent in transfer window?
    Transfer window sentiment?
    Star player playing/missing?
    
    To evaluate the model I will be using the following metrics:
        Accuracy (This should be fine as all classes are equally represented and there is no cost to misclassification)
        Look a probability calibration to see if the model is over or under confident
        Look at plotting ROC and AUC to see how well the model is performing
        look at outputting some sort of confidence in the prediction. So I know which predictions the model is most confident in.
        To backtest, I will need to access previous odds data. I will then use this to see if the model is profitable. (https://the-odds-api.com/)
        Possibly also look at a poisson distributin for calculating how many goals will be scored in a game.
        I will also use SHAPELY values to see which features are most important in the model.
    
    Finally, I will ceate a web app to display the precitions for the games that weekend, the confidence in the predictions, and the PnL for the season. 
    It will also display the backtested PnL. (May have to pay for historical odds data)

"""
# Setting up the directories for the project
from pathlib import Path
import os
import pandas as pd

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
