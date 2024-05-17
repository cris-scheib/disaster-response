# Disaster Response Pipeline Project


This project aims to creating a machine learning pipeline to categorize evens using real messages that were sent during disaster events. This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

Below are a few screenshots of the web app.


Dashboard        | Classifier
:---------------|-------------------:
![image](https://github.com/cris-scheib/disaster-response/assets/61483993/df4461ed-685a-4432-8c87-ab22ee6a28ca) | ![image](https://github.com/cris-scheib/disaster-response/assets/61483993/9562c133-d609-4176-9c82-a5d6fceeda63)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Project Components
There are three components you'll need to complete for this project.

#### ETL Pipeline
In a Python script, process_data.py, we have the data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

#### ML Pipeline
In a Python script, train_classifier.py, we have the machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

#### Flask Web App
In the flask web app we implement a interface and display the results to the user. 

