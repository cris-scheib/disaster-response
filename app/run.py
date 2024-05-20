import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
import sqlite3


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
connection = sqlite3.connect('../data/DisasterResponse.db')
df = pd.read_sql_query("SELECT * FROM data_table", connection)
connection.close()

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphsPlot1 = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    dist_values = dict()
    for column in df.iloc[:, 4:].columns.tolist():
        dist_values[column] = (df[column] == 1).sum()

    categories = list(dist_values.keys())
    categories_counts = list(dist_values.values())

    graphsPlot2 = [
        {
            'data': [
                Bar(
                    x=categories,
                    y=categories_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories Type',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Type"
                }
            }
        }
    ]
    # encode plotly graphs in JSON
    graphsPlot1JSON = json.dumps(graphsPlot1, cls=plotly.utils.PlotlyJSONEncoder)
    graphsPlot2JSON = json.dumps(graphsPlot2, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', graphJSON=[graphsPlot1JSON, graphsPlot2JSON])


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
