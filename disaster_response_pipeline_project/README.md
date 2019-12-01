# Disaster Response Pipeline Project

This project is part of Udacity Data Scientist Nanodegree program. 

## Table of Contents
1. [Project Motivation](#motivation)
2. [File Description](#files)
3. [Instructions](#instructions)

## Project Motivation <a name="motivation"></a>
This is an Udacity Nanodegree project. In this project, a web-application is built. This web-application, with pre-trained messages and optimized classifier can predict response category of newcoming messages. This categorization will help in reducing reaction time of the responding organizations.</br>

## File Description <a name="files"></a>
- 'app/'
  - 'template/'
    - 'master.html'  -  Main page of web application.
    - 'go.html'  -  Classification result page of web application.
  - 'run.py'  - Flask applications main file.

- 'data/'
  - 'disaster_categories.csv'  - Disaster categories dataset.
  - 'disaster_messages.csv'  - Disaster Messages dataset.
  - 'process_data.py' - The data processing pipeline script.

- 'models/'
  - 'train_classifier.py' - The NLP and ML pipeline script.

    
 - 'preparation/' 
  - 'ETL Pipeline Preparation.ipynb'  -  Jupyter notebook for ETL preparation
  - 'ETL Pipeline Preparation.html'  -  HTML version of Jupyter notebook for ETL preparation
  - 'ML Pipeline Preparation.ipynb'  -  Jupyter notebook for ML preparation
  - 'ML Pipeline Preparation.html'  -  HTML version of Jupyter notebook for ML preparation
  - 'messages.csv'  -  CSV file containing the messages
  - 'categories.csv'  -  CSV file containing the categories
</br>

### Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
    - To run ML pipeline that trains classifier and saves
        'python models/train_classifier.py data/DisasterResponse.db models/disaster_resp_classifier.pkl'

2. Run the following command in the app's directory to run the web app.
    'python run.py'

3. Go to http://0.0.0.0:3001/ or http://localhost:3001
