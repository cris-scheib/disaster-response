import sys
import pandas as pd
import sqlite3
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


def load_data(database_filepath):
    '''
    returns data loaded from a database
    
    input:
        database_filepath: the filepath of the database
    
    output:
        X: input data 
        Y: target variable 
        categories: a list of the categories 
    '''
    connection = sqlite3.connect(database_filepath)
    df = pd.read_sql_query("SELECT * FROM data_table", connection)
    connection.close()
    X = df['message']
    Y = df.iloc[:, 4:]

    dist_cat = dict()
    for column in Y.columns.tolist():
        dist_cat[column] = (df[column] == 1).sum()
    categories = list(dist_cat.keys())

    return X, Y, categories


def tokenize(text):
    '''
    returns the tokenized text
    
    input:
        text: the phrase to be tokenized
    
    output:
        clean_tokens: a list of tokens 
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    returns the model build in a Pipeline
    
    output:
        pipeline: the pipeline of the model 
    '''
    return Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('multi_clf', MultiOutputClassifier(
            RandomForestClassifier(n_estimators=300)))
    ])


def evaluate_model(model, X_test, Y_test, categories):
    '''
    prints the evaluation of the model
    
    input:
        model: the model
        X_test: input data of the test dataset
        Y_test: target variable of the test dataset
        categories: a list of the categories 
    '''
    y_pred = model.predict(X_test)
    y_true = Y_test.to_numpy()

    # Calculate precision, recall, and F1 score for each output variable separately
    precision = []
    recall = []
    f1 = []

    num_output_variables = Y_test.shape[1]  # Assuming y_true and y_pred have the same shape

    for i in range(num_output_variables):
        precision.append(precision_score(y_true[:, i], y_pred[:, i], average='micro'))
        recall.append(recall_score(y_true[:, i], y_pred[:, i], average='micro'))
        f1.append(f1_score(y_true[:, i], y_pred[:, i], average='micro'))

    # Print or store the results
    for i in range(num_output_variables):
        print(f"Output Variable {categories[i]}:")
        print(f"Precision: {precision[i]}")
        print(f"Recall: {recall[i]}")
        print(f"F1 Score: {f1[i]}")
        print()


def save_model(model, model_filepath):
    '''
    saves the model in a pickle file
    
    input:
        model: the model to be save
        model_filepath: the filepath to save the model
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, categories = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categories)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
