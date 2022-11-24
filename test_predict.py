

import pandas as pd
import numpy as np

from config.base_conf import *
from func import preprocess_for_optimize
from models.linear_opt import *


def main():
    industry = pd.read_csv(RAW_DATA_PATH+ '/Industry data.csv')
    retail = pd.read_csv(RAW_DATA_PATH + '/Retail data.csv')
    profit = pd.read_csv(PREPROCESSED_DATA + '/item_info.csv')

    retail_month = extract_month(retail, '2022-03', '2022-04')

    ind_const = val_constraint(industry, '2022-03', '2022-04')
    model, item_tag = LinearOpt(ind_const, retail_month, profit, 3333)

    model = run_model(model)

    result_df = submission(model, item_tag)

    result_df.to_csv(OUTPUT_PATH+'/submission_2.csv', index=False)


if __name__ == '__main__':
    main()









