import json
import plotly
import pandas as pd
import nltk
nltk.download(['stopwords'])

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Layout, Figure
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
	"""
	Tokenizes the text
	Args:
	text: text that needs to be tokenized
	Returns:
	tokens: list of tokens
	"""
	#sanitize - replace non-alphanumeric char with space, and normalize
	text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

	#remove stop words
	words = word_tokenize(text)
	words = [w for w in words if w not in stopwords.words("english")]

	#lemmatize
	lemmatizer = WordNetLemmatizer()
	tokens = []
	for word in words:
		token = lemmatizer.lemmatize(word).lower().strip()
		tokens.append(token)

	return tokens


def messages_by_categories(df, count_of_categories):
    """Create a graph for messages per category
    Args:
    df: the dataframe
    count_of_categories: number of categories to be included
    Returns:
    figure: graph of messages per category
    """

    # Group by categories
    categories = df.iloc[:, 4:].sum().sort_values(ascending=False)

    data = [Bar(
        x=categories.index,
        y=categories,
        opacity=0.8
    )]

    layout = Layout(
        title="Messages per category",
        xaxis=dict(
            title='Categories',
            tickangle=45
        ),
        yaxis=dict(
            title='Number of messages',
        )
    )

    cat_end_index = count_of_categories + 1
    figure = Figure(data=data, layout=layout), categories.index[:cat_end_index]
    
    return figure


def categories_per_genre(df, categories, count_of_categories):
    """Create a graph for categories per genre
    Args:
    df: the dataframe
    categories: categories
    count_of_categories: number of categories to be included
    Returns:
    figure: graph of categories per genre
    """
    # Group by genres
    genres = df.groupby('genre').sum()[categories]

    color_bar = 'DarkGreen'

    data = []
    for cat in genres.columns[1:]:
        data.append(Bar(
                    x=genres.index,
                    y=genres[cat],
                    name=cat)
                    )

    layout = Layout(
        title="Categories per genre (Top " + str(count_of_categories) + ")",
        xaxis=dict(
            title='Genres',
            tickangle=45
        ),
        yaxis=dict(
            title='number of messages per Category',
        )
    )

    figure = Figure(data=data, layout=layout)

    return figure


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/disaster_resp_classifier.pkl")

# create graphs
count_of_categories = 10
msgs_per_cat_graph, top_categories = messages_by_categories(df, count_of_categories)
cats_per_genre_graph = categories_per_genre(df, top_categories, count_of_categories)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # encode plotly graphs in JSON
    graphs = [msgs_per_cat_graph, cats_per_genre_graph]
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()