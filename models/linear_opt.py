
import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpAffineExpression, LpElement

from config.base_conf import *
from func.preprocess_for_optimize import *

# for validation
def val_constraint(industry, start_date, end_date):
    ind_month = extract_month(industry, start_date,end_date)
    item_tag2 = item_tag_2()
    ind_month_2 = adding_store_product_var(ind_month, item_tag2)
    industry_const = ind_month_2[['Industry Size','var']]

    ind_const = {}
    for i in range(industry_const.shape[0]):
        ind_const[industry_const.iloc[i][1]] = industry_const.iloc[i][0]
    
    return ind_const 


# for validation
def LinearOpt(ind_const, retail_month, profit, N=None):
    item_tag = item_tag_1(profit)
    if N == None:
        N = cal_N(retail_month, profit)
    #define model
    model = LpProblem(sense = LpMaximize)
    #define x variable names (x_name = [1_0, 1_1, ..642_17])
    # 이 x_name이 모든 변수여야 함.-> range함수써서 다 만들자 
    # ind_const['var'].tolist()은 그 시기에 매입한 상점과 상품 수 
    x_name = list(ind_const.keys())
    x = [LpVariable(x_name[i], lowBound=0, upBound=ind_const[x_name[i]]) for i in range(len(x_name))]
    #obj function
    obj_func = [(x[i] ,item_tag[int(str(x[i]).split('_')[1])][1]) for i in range(len(x_name))]
    obj_function = LpAffineExpression(obj_func)

    model += (np.sum(x) <= N)
    model += obj_function
    print("N : " , N)
    return model , item_tag

def run_model(model):
    status = model.solve()
    print(f"status: {model.status}, {LpStatus[model.status]}")
    print(f'Objective : {model.objective.value()}')
    return model 

#for validation
def submission(model, item_tag):
    result = list()
    for var in model.variables():
        result.append([var.name, var.value()])
    result = pd.DataFrame(result, columns = ['name','value'])

    result['store'] = result['name'].apply(lambda x : int(x.split('_')[0]))
    result['product'] = result['name'].apply(lambda x : int(x.split('_')[1]))

    result = result.sort_values(by=['store', 'product'])
    # result['Item'] = result['product'].apply(lambda x : item_tag[x][0])
    result = result[['store','product','value']]
    result['value'] = result['value'].astype(int)
    result = result.reset_index()
    result.drop(['index'], axis = 1, inplace=True)

    # 전체 변수 넣어주기
    # 빈 df 만들기
    merge_into = list()
    for i in range(1, 643):
        for j in range(18):
            merge_into.append([i,j,0])
    
    merge_into = pd.DataFrame(merge_into)
    merge_into.columns = ['store','product','value']

    result_df = pd.merge(merge_into, result, how='outer', on=['store','product'])


    def remove_na(val):
        if pd.isnull(val):
            return 0
        else:
            return val
    
    result_df['value_y'] = result_df['value_y'].apply(remove_na)
    result_df.drop(['value_x'], axis = 1, inplace=True)
    result_df['value_y'] = result_df['value_y'].astype(int) 
    result_df['Item'] = result_df['product'].apply(lambda x : item_tag[x][0])
    #여기까지 result_df column은 store, product, value_y, Item
    result_df = result_df[['store','Item','value_y']]
    result_df.columns = ['Store','Item','Allocation']

    return result_df 



    


