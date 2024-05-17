import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    returns data loaded from csv files
    
    input:
        messages_filepath: the filepath of the messages.csv
        categories_filepath: the filepath of the categories.csv
    
    output:
        df: input data 
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')

def clean_data(df):
    '''
    returns clean data from a dataframe
    
    input:
        df: the original database
    
    output:
        df: clean and processed dataframe 
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = list(map(lambda x: x[:-2], row)) 
    categories.columns = category_colnames

    for column in categories:
        categories[column] = list(map(lambda x: int(x[-1]), categories[column]))

    df = df.drop('categories', axis=1)
    df = df.join(categories)
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    '''
    saves the dataframe in a dump
    
    input:
        df: the dataframe to be save
        database_filename: the filepath to save the dump
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('data_table', engine, index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()