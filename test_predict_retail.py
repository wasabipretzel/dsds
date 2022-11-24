
import pandas as pd
import numpy as np

from config.base_conf import *
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpAffineExpression, LpElement
from func import preprocess_for_optimize
from models.linear_opt import *

# retail data로 optimization -> 문제는 10K일 때 해당 기간 retail 실제 배치 수가 
# 더 적어서 개선이 안됨. -> 더 할당가능한 부분을 해결 위해, 전체 변수로 하여,
# 할당 가능한 부분을 industry에서 비율로 가져오되, 만약 그것도 없다면 
# 가장 물량을 많이 매입하는 store, 가장 가까운 매장으로 할당하자

def main():
    N=3333

    industry = pd.read_csv(RAW_DATA_PATH+ '/Industry data.csv')
    retail = pd.read_csv(RAW_DATA_PATH + '/Retail data.csv')
    profit = pd.read_csv(PREPROCESSED_DATA + '/item_info.csv')

    # val constraint
    retail_month = extract_month(retail, '2022-03','2022-04')

    item_tag2 = item_tag_2()
    retail_month_2 = adding_store_product_var(retail_month, item_tag2)
    retail_const = retail_month_2[['Sales Total','var']]

    re_const = {}
    for i in range(retail_const.shape[0]):
        re_const[retail_const.iloc[i][1]] = retail_const.iloc[i][0]
    
    # re_const : use for constraint

    #Linear opt
    item_tag = item_tag_1(profit)

    model = LpProblem(sense = LpMaximize) 
    x_name = list(re_const.keys())
    x = [LpVariable(x_name[i], lowBound=0, upBound=re_const[x_name[i]]) for i in range(len(x_name))]
    #obj function
    obj_func = [(x[i] ,item_tag[int(str(x[i]).split('_')[1])][1]) for i in range(len(x_name))]
    obj_function = LpAffineExpression(obj_func)

    model += (np.sum(x) <= N)
    model += obj_function
    
    model = run_model(model)
    
    result_df = submission(model, item_tag)
    result_df.to_csv(OUTPUT_PATH+'/retail_sub_3333.csv', index=False)


    


if __name__ == '__main__':
    main()
