import sys
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
	"""
	Loads messages and categories from input parameters, and merges them into a dataframe
	Args:
	messages_filepath: CSV file containing messages
	categories_filepath: CSV file containing categories
	Returns:
	df: Dataframe obtained from merging the two inputs
	"""
	# load messages dataset
	messages = pd.read_csv(messages_filepath)
	# load categories dataset
	categories = pd.read_csv(categories_filepath)
	# merge datasets
	df = messages.merge(categories, on='id')
	print('Messages and categories have been loaded successfully.')
	
	return df


def clean_data(df):
	"""
	Cleans the dataframe
	Args:
	df: dataframe to be cleaned
	Returns:
	df: Cleaned dataframe
	"""
	# create a dataframe of individual category columns
	categories = df['categories'].str.split(';', expand=True)
	# select the first row of the categories dataframe
	row = categories.iloc[0].values
	# use this row to extract a list of new column names for categories.
	# one way is to apply a lambda function that takes everything 
	# up to the second to last character of each string with slicing
	extract_col_names = lambda x: [r[:-2] for r in x]
	category_colnames = extract_col_names(row)
	# rename the columns of `categories`
	categories.columns = category_colnames
	
	for column in categories:
		# set each value to be the last character of the string
		categories[column] = categories[column].str[-1]
		# convert column from string to numeric
		categories[column] = pd.to_numeric(categories[column])
		
	# drop the original categories column from 'df'
	df.drop(['categories'], axis=1, inplace=True)
	# concatenate the original dataframe with the new 'categories' dataframe
	df[categories.columns] = categories
	# drop duplicates
	df.drop_duplicates(inplace = True)
	print('Dataframe has been cleaned successfully.')
	
	return df


def save_data(df, database_filename):
	"""
	Saves dataframe into a database
	Args:
	df: dataframe to be saved
	database_filename: file path of database
	"""
	# remove existing database, if exists
	if os.path.exists(database_filename):
		os.remove(database_filename)
		print('Removed existing DB: ' + database_filename)
		
	# create new database and save there
	engine = create_engine('sqlite:///' + database_filename)
	df.to_sql('DisasterResponse', engine, index=False)


def main():
	if len(sys.argv) == 4:

		messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

		print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
		df = load_data(messages_filepath, categories_filepath)

		print('Cleaning data...')
		df = clean_data(df)
		
		print('Saving data...\n    DATABASE: {}'.format(database_filepath))
		save_data(df, database_filepath)
		
		print('Cleaned data saved to database!')

	else:
		print('Please provide required number of arguments to the program. '\
			'\nThe program needs to be executed as follows: '\
			'\npython process_data.py {messages CSV} {categories CSV} {database} '\
			'\n\nExample: python process_data.py '\
			'disaster_messages.csv disaster_categories.csv '\
			'DisasterResponse.db')


if __name__ == '__main__':
	main()