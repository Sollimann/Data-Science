"""
 Written by Kristoffer Rakstad Solberg,
 Summer Intern, AKBM Transformation,
 2019,
"""

import math
import numpy as np
from datetime import timedelta
from geopy.distance import great_circle
"""
SOURCE:
    Birant, D. and Kut, A. (2007). St-dbscan: An algorithm for clustering 
    spatial–temporal data. Data & Knowledge Engineering, 60(1):208 – 221. 
    Intelligent Data Mining.

INPUTS:
    df={o1,o2,...,on} Set of objects
    spatial_threshold = Maximum geographical coordinate (spatial) distance value
    temporal_threshold = Maximum non-spatial distance value
    min_neighbors = Minimun number of points within Eps1 and Eps2 distance
OUTPUT:
    C = {c1,c2,...,ck} Set of clusters
"""
def ST_DBSCAN(df, spatial_threshold, temporal_threshold, min_neighbors):
    cluster_label = 0
    NOISE = -1
    UNMARKED = 777777
    stack = []

    # initialize each point with unmarked
    df['cluster'] = UNMARKED
    
    # for each point in database
    for index, point in df.iterrows():
        if df.loc[index]['cluster'] == UNMARKED:
            neighborhood = retrieve_neighbors(index, df, spatial_threshold, temporal_threshold)
            
            if len(neighborhood) < min_neighbors:
                df.set_value(index, 'cluster', NOISE)

            else: # found a core point
                cluster_label = cluster_label + 1
                df.set_value(index, 'cluster', cluster_label)# assign a label to core point

                for neig_index in neighborhood: # assign core's label to its neighborhood
                    df.set_value(neig_index, 'cluster', cluster_label)
                    stack.append(neig_index) # append neighborhood to stack
                
                """
                    The stack is necessary to find density-reachable objects from
                    directly density-reachable objects
                """
                while len(stack) > 0: # find new neighbors from core point neighborhood
                    current_point_index = stack.pop()
                    
                    """
                    mabye add filtering of months here?
                    months = get_re
                    """
                    
                    new_neighborhood = retrieve_neighbors(current_point_index, df, spatial_threshold, temporal_threshold)
                    
                    if len(new_neighborhood) >= min_neighbors: # current_point is a new core
                        for neig_index in new_neighborhood:
                            neig_cluster = df.loc[neig_index]['cluster']
                            if (neig_cluster != NOISE) & (neig_cluster == UNMARKED): 
                                # TODO: verify cluster average before add new point
                                df.set_value(neig_index, 'cluster', cluster_label)
                                stack.append(neig_index)
    return df

def get_relevant_months(date,timedelta):

    """
    # earliest month
    if (timedelta > month):
        month_earliest = (12-timedelta) + month
    else:
        month_earliest = month - timedelta
    
    # latest month
    month_latest =  month + timedelta
    if (month_latest) > 12:
        month_latest -= 12
    """ 
    
    # init
    months = []
    months.append(date)
    
    # add months
    for month in range(1, timedelta+1):
        
        if (date + month) > 12:
            #wrap month
            months.append(date + month - 12)
            months.append(date - month)
        elif (date - month) < 1:
            # wrap month
            months.append(date - month + 12)
            months.append(date + month)
        else:
            months.append(date + month)
            months.append(date - month)
    
    # might add sorting here
    return months

def retrieve_neighbors(index_center, df, spatial_threshold, temporal_threshold):
    neigborhood = []

    center_point = df.loc[index_center]

    # filter by time 
    #min_time = center_point['month'] - timedelta(days = temporal_threshold)
    #max_time = center_point['month'] + timedelta(days = temporal_threshold)
    
    months = get_relevant_months(center_point['month'],temporal_threshold)
    df = df[np.isin(df['month'],months)]

    # filter by distance
    for index, point in df.iterrows():
        if index != index_center:
            distance = great_circle((center_point['Latitude'], center_point['Longitude']), (point['Latitude'], point['Longitude'])).km
            if distance <= spatial_threshold:
                neigborhood.append(index)

    return neigborhood