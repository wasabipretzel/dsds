

import pandas as pd 
import numpy as np 

from config.base_conf import * #<- for debugging

def item_tag_1(profit_df):
	"""Summary
	
	Args:
	    profit_df (dataframe): profit per product.csv in raw folder
	
	Returns:
	    dict: dictionary matching {integer : [product name, profit of product], ...}
	"""
	item_order = ['Bracket','Brush','Cereal','Ear Buds','Eggs','Glue','Keyboard','King Crab','Milk','Mouse','Nails','Noodles','Paint','Phone Charger','Power Cord','Shrimp','Steak','Tape']
	
	item_profit = dict()

	for i in range(profit_df.shape[0]):
		item_profit[profit_df['Item'].iloc[i]] = profit_df['Profit'].iloc[i]
	
	item_tag_1 = dict()
	for i, items in enumerate(item_order):
		item_tag_1[i] = [items, item_profit[items]]
	
	return item_tag_1
	
def item_tag_2():
	"""Summary
	
	Returns:
	    dict: dictionary matching {product name : integer}
	"""
	item_order = ['Bracket','Brush','Cereal','Ear Buds','Eggs','Glue','Keyboard','King Crab','Milk','Mouse','Nails','Noodles','Paint','Phone Charger','Power Cord','Shrimp','Steak','Tape']
	item_tag_2 = dict()
	for i, items in enumerate(item_order):
		item_tag_2[items] = i  
	
	return item_tag_2

#for validation
def extract_month(sales_data, start_month, end_month):
	"""_summary_
		extracting sub dataframe from start_month before end_month
		e.g) extract_month(industry_data, '2021-01', '2021-02') : extract data of 2021-01-01 ~ 2021-01-31
	Args:
		sales_data (dataframe):  industry data or retail data
		start_month (string): year and month want to start extract. e.g.) '2021-01'
		end_month (string): year and month want to stop extract. e.g) '2021-02'

	Returns:
		dataframe: sub dataframe of sales_data
	"""
	return sales_data[(sales_data['Month'] >= '{}-01 00:00:00'.format(start_month)) & (sales_data['Month'] < '{}-01 00:00:00'.format(end_month))]

#for vaildation
def cal_N(retail_month, profit_df):
	"""Summary
	
	Args:
	    retail_month (dataframe): extracted specific month retail data
	    profit_df (dataframe): profit per product.csv in raw folder
	
	Returns:
	    int: sum of retail data. Sum of supplier's allocation
	"""
	merged_product_profit = pd.merge(retail_month, profit_df[['Item','Profit']], how='left', left_on='Item', right_on='Item')
	return merged_product_profit['Sales Total'].mul(merged_product_profit['Profit']).sum()

def adding_store_product_var(industry_month, item_tag_2):
	"""_summary_
		adding 'var' column which represent 'storenumber_productnumber'. 
	Args:
		industry_month (dataframe): sub dataframe extracted by extract_month
		item_tag_2 (dict): dictionary made from item_tag_2()
	"""
	def match(item_name):
		return item_tag_2[item_name]

	industry_month['Item_var'] = industry_month['Item'].apply(match)
	industry_month['var'] = industry_month.apply(lambda row : str(row.Store) + "_" + str(row.Item_var), axis = 1)

	return industry_month

