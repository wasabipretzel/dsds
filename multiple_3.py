
import pandas as pd

from config.base_conf import *


def main():
    base = pd.read_csv(OUTPUT_PATH + '/retail_sub_3333.csv')
    base['real_allo'] = base['Allocation'] * 3

    base.drop(['Allocation'], axis = 1, inplace=True)
    base.columns=['Store','Item','Allocation']
    base.to_csv(OUTPUT_PATH + '/retail_10000_allocation.csv', index=False)


if __name__ == '__main__':
    main()