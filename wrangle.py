import pandas as pd
import numpy as np
import os

# acquire
from env import get_db_url
from pydataset import data
import seaborn as sns

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

def wrangle_zillow():
    '''
    This function reads the zillow data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
                FROM properties_2017
                WHERE propertylandusetypeid = 261
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('zillow'))
    # replace any whitespace with null values
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    # drop out any null values:
    df = df.dropna()
    # cast everything as an integer:
    df = df.astype(int)
    
    return df