import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def missing_data(df):
    """ This function takes in a dataframe, calculates
    the total missing values and the percentage of missing
    values per column and returns a new dataframe.
    """
    new = pd.DataFrame()
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().mean()*100).sort_values(ascending = False)
    new = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return new.loc[new['Percent']>0]