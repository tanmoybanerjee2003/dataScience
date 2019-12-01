import sys
import os
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_recall_fscore_support
import pickle
import warnings

# to ignore warnings
warnings.filterwarnings("ignore")


def load_data(database_filepath):
	"""
	Loads data from database
	Args:
	database_filepath: path to the database
	Returns:
	X: Features dataframe
	Y: Target dataframe 
	"""
	# load data from database
	engine = create_engine('sqlite:///' + database_filepath)
	df = pd.read_sql_table('DisasterResponse', con = engine)
	X = df['message']
	Y = df.iloc[:,4:]
	print('Data have been successfully loaded from the database.')
	
	return X, Y


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


def build_model():
	"""
	Builds classification model.
	Returns:
	cv: the classifier model
	"""
	pipeline = Pipeline([
		('vectorizer', CountVectorizer(tokenizer=tokenize)),
		('tfidf', TfidfTransformer()),
		('clf', MultiOutputClassifier(OneVsRestClassifier(RandomForestClassifier())))])
		
	# Use grid search to find better parameters.
	parameters = {
		'vectorizer__ngram_range': ((1, 1), (1, 2)),
		'tfidf__use_idf': [True, False]
	}
	# In Windows, spawning parallel threads for grid search might encounter error.
	# In that case, invoke grid search in single-thread model i.e. comment following line and uncomment next line.
	cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1, n_jobs=-1)
	#cv = GridSearchCV(pipeline, param_grid=parameters)
	print('Model has been built successfully.')
	
	return cv


def compute_results(y_test, y_pred):
    """
    Computes f-score, precision and recall based on predicted and test labels
    Args:
    y_test: test labels
    y_pred: predicted lables
    Returns:
    results: dataframe containing category-wise results
    aggregated_results: dataframe containing aggregated results
    """
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    for category in y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[category], y_pred[:,num], average='weighted')
        results.set_value(num+1, 'Category', category)
        results.set_value(num+1, 'f_score', f_score)
        results.set_value(num+1, 'precision', precision)
        results.set_value(num+1, 'recall', recall)
        num += 1
    
    aggregated_results = pd.DataFrame(columns=['Aggregated_Result_type', 'Value'])
    line = pd.DataFrame({'Aggregated_Result_type': 'f_score', 'Value': results['f_score'].mean()}, index=[0])
    aggregated_results = aggregated_results.append(line)
    line = pd.DataFrame({'Aggregated_Result_type': 'precision', 'Value': results['precision'].mean()}, index=[1])
    aggregated_results = aggregated_results.append(line)
    line = pd.DataFrame({'Aggregated_Result_type': 'recall', 'Value': results['recall'].mean()}, index=[2])
    aggregated_results = aggregated_results.append(line)
    
    return results, aggregated_results


def evaluate_model(model, X_test, Y_test):
	"""
	Evaluate the model
	Args:
	model: Trained model
	X_test: Test features
	Y_test: Test labels 
	"""
	# predict
	Y_pred = model.predict(X_test)
	# display results
	results, aggregated_results = compute_results(Y_test, Y_pred)
	print(aggregated_results)
	print('Evaluation of the model has been completed successfully.')


def save_model(model, model_filepath):
	"""
	Saves the model to a Python pickle file    
	Args:
	model: model to be saved
	model_filepath: pickle file where the model will be saved
	"""
	# remove existing pickle file, if exists
	if os.path.exists(model_filepath):
		os.remove(model_filepath)
		print('Removed existing pickle file: ' + model_filepath)
		
	# save into new pickle file
	pickle.dump(model, open(model_filepath, 'wb'))
	print('Model has been saved into pickle file successfully !')


def main():
	if len(sys.argv) == 3:
		database_filepath, model_filepath = sys.argv[1:]
		print('Loading data...\n DATABASE: {}'.format(database_filepath))
		X, Y = load_data(database_filepath)
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 40)

		print('Building model...')
		model = build_model()

		print('Training model...')
		model.fit(X_train, Y_train)
		print('Model has been trained successfully.')

		print('Evaluating model...')
		evaluate_model(model, X_test, Y_test)

		print('Saving model...\n MODEL: {}'.format(model_filepath))
		save_model(model, model_filepath)

	else:
		print('Please provide required number of arguments to the program. '\
			'\nThe program needs to be executed as follows: '\
			'\npython train_classifier.py {database to be read} {pickle file to be created}'\
			'\n\nExample: python train_classifier.py ../data/DisasterResponse.db disaster_resp_classifier.pkl')


if __name__ == '__main__':
	main()