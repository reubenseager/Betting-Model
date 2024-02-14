"""
    Here we will create a pipeline for the prediction of the model. THere will be no training done here, only prediction.
    
    The pipeline will be as follows:
        - Read in the data (This includes: Webscraped match data, betting data (historical + live), ELO data)
        - Preprocessing the data (This includes creating features, selecting the important ones, scaling etc)
        - Predicting the results of the matches (The results will then be predicted for the upcoming games using the pretrained level 1 classifier)
        - Saving the predictions to a file
        - Ongoing evaluation of the models performance
        - Display data on a public webapp dashboard (This will be hosted on AWS and will be created using streamlit)
    
"""


