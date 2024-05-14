import sys
import pandas as pd
import numpy as np
import sqlite3
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


def load_data(database_filepath):
    connection = sqlite3.connect(database_filepath)
    df = pd.read_sql_query("SELECT * FROM data_table", connection)
    connection.close()
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    return Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('multi_clf', MultiOutputClassifier(
            RandomForestClassifier(n_estimators=300)))
    ])


def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    accuracy = (y_pred == Y_test).mean()

    print("Accuracy:", accuracy)
    pass


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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
