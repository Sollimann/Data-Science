#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from datetime import timedelta 
import pandas as pd
import numpy as np
import os
import time
import warnings
import json
from datetime import datetime, timedelta
#from netCDF4 import Dataset
#from scipy.interpolate import griddata


# In[22]:


import sys 
sys.path.append('/home/jupyter/catch/route/') # setting sys path to include project-root-folder
from utils.download_catch_data_BigQuery import extract_catch_data_BigQuery


# In[25]:


def load_BQ_and_clean():
    '''
    The 'load_BQ_and_clean()' function calls 'extract_catch_data_BigQuery()' and extracts AKBM catch data and does some additional cleaning. The output 

    '''
    config_path = 'analytics-secure-pillar-201009-serviceaccount_keyfile.json' # NOTE: hardcoded! 
    catch_data = extract_catch_data_BigQuery(config_path, config_dict=None, savefile=False, verbose=1)

    # turning off warnings
    pd.options.mode.chained_assignment = None  # default='warn'

    # Cleaning
    # fix wrong value (hit 0 instead of .)
    catch_data['Krill Size (mm)'].replace(to_replace=[3807.0],value=38.07, inplace=True)
    # ten times too large?
    catch_data['Krill weight (gram)'].replace(to_replace=[3.63],value=0.363, inplace=True)

    # setting temp weight's and getting index for data with switched columns
    idx = catch_data[(catch_data['Krill weight (gram)']>10) & (catch_data['Krill Size (mm)']<1)]['Krill weight (gram)'].index
    temp_size = catch_data[(catch_data['Krill weight (gram)']>10) & (catch_data['Krill Size (mm)']<1)]['Krill weight (gram)']
    temp_weight = catch_data[(catch_data['Krill weight (gram)']>10) & (catch_data['Krill Size (mm)']<1)]['Krill Size (mm)']
    # setting correct values
    catch_data['Krill weight (gram)'][idx.values]=temp_weight
    catch_data['Krill Size (mm)'][idx.values]=temp_size

    # idx_corrected_weight = catch_data[(catch_data['Krill weight (gram)']>10)]['Krill weight (gram)'].index
    corrected_weight = catch_data[(catch_data['Krill weight (gram)']>10)]['Krill weight (gram)']/100
    idx_corrected_weight = corrected_weight.index
    catch_data['Krill weight (gram)'][idx_corrected_weight]=corrected_weight

    # Getting the description
    description=catch_data.describe()
    # setting Krill size less than 10 mm to the mean Krill size. 
    catch_data[(catch_data['Krill Size (mm)']<10)]['Krill Size (mm)']=description['Krill Size (mm)'][1] # Does not work?
    
    return catch_data

