import pandas as pd 

from config.base_conf import *
from func import preprocess_for_optimize as po

industry = pd.read_csv(RAW_DATA_PATH+ '/Industry data.csv')
retail = pd.read_csv(RAW_DATA_PATH + '/Retail data.csv')
profit = pd.read_csv(PREPROCESSED_DATA + '/item_info.csv')

ind_month = po.extract_month(industry, '2021-01','2021-02')
retail_month = po.extract_month(retail, '2021-01', '2021-02')
N = po.cal_N(retail_month, profit)
item_tag_1 = po.item_tag_1(profit)
item_tag_2 = po.item_tag_2()
ind_month_2 = po.adding_store_product_var(ind_month, item_tag_2)

print(ind_month.head())
print(retail_month.head())
print(N)
print(item_tag_1)
print(item_tag_2)
print(ind_month_2)