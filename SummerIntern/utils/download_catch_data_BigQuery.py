"""
Functions to download/extract catch data from BigQuery.

Note that you need to specify a config (.json) file to establish a connection to Bigquery,
see help(load_credentials_GCP)
"""
import json
import pandas as pd
import time
import os
from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud.exceptions import BadRequest

# Import custom functions
from utils.preprocessing_AKBM_catch_data import clean_catch_data


class BigQueryClient():
    """
    Class for establishing a client connection to BigQuery and extract/load data.
    """
    def __init__(self, config_path, config_dict=None):
        """
        Initiate BigQuery client object, for this a config file is needed

        json-file in config_path should have fields:
            {
              "type": "service_account",
              "project_id": [project_id],
              "private_key_id": [private_key_id],
              "private_key": "-----BEGIN PRIVATE KEY-----[private_key]==\n-----END PRIVATE KEY-----\n",
              "client_email": [client_email],
              "client_id": [client_id],
              "auth_uri": [auth_uri],
              "token_uri": [token_uri],
              "auth_provider_x509_cert_url": [auth_provider],
              "client_x509_cert_url": [url]
            }

        Arguments:
        ----------
            config_path -- string
                path to service account .json file
            config_dict -- dictionary, default None
                Optional input - send as argument the actual json content as a dict
        """
        if config_dict is not None:
            creds = config_dict
        else:
            with open(config_path, "r") as creds_file:
                creds = json.load(creds_file)

        self.project_id = creds['project_id']

        # Establish connection
        credentials = service_account.Credentials.from_service_account_file(config_path)
        self.client = bigquery.Client(credentials=credentials, project=self.project_id)


def extract_catch_data_BigQuery(config_path, config_dict=None, savefile=True, verbose=1):
    """
    Function to extract data from BigQuery on GCP, in particular the catch data that has been
    uploaded by Cuba and  Sii team based on Box production reports (daily `.xlsx` files).

    The tables/views that are extracted, cleaned and merged are:
        - `akbm_vessels_dev_productionReporting.Catch_Historic`   (contains years 2010-2018)
        - `akbm_vessels_dev_productionReporting.BridgeReporting`  (position data, starting December 2018)
        - `akbm_vessels_dev_productionReporting.FactoryReporting` (catch volume data, starting December 2018)

    Notes:
        - In order to establish the connection with MySQL in GCP (i.e. in order to run this function)
          one first need to initiate a cloud_sql_proxy script.

    Arguments:
    ----------
        config_path -- string, path to config file with MySQL credentials, typically in '../config/XXX.json'
        config_dict -- config file content as dict (optional to sending config path)
        savefile -- boolean, default True
            if True then save merged and cleaned dataset to '../data/All_catch_data_to_[enddate].csv'
        verbose -- int, in [0: do not print runtime info, 1: print runtime info], default 1

    Returns:
    --------
        df_catch -- pandas DataFrame with catch data
    """

    def extract_data_to_102018(client):
        """
        Extract data from MySQL up to Octobr 2018; this data is in the table prod.CatchHistoric
        """
        query_2010to2018 = """ 
            SELECT Vessel, ProductionDay, TotalCatchKrillMt, TotalKrillMealKg,
                    Position, Area, TrawlDepthM, BaricPressure, WaterTempDegreeC, Wind,
                    KrillSizeMm, KrillWeightGram, KrillSizeMmPerGram, Comments
            FROM akbm_vessels_dev_productionReporting.Catch_Historic  
        """
        df_to2018 = client.query(query_2010to2018).to_dataframe()

        # Rename some columns
        df_to2018.rename(index=str, columns={"ProductionDay": "Date", "TotalCatchKrillMt": "Total catch Krill - Mt",
                                             "TotalKrillMealKg": "Total Krill Meal Kg", "TrawlDepthM": "Trawl depth (m)",
                                             "BaricPressure": "Baric pressure (hPa)",
                                             "WaterTempDegreeC": "Water temp (Celsius)",
                                             "KrillSizeMm": "Krill Size (mm)", "KrillWeightGram": "Krill weight (gram)",
                                             "KrillSizeMmPerGram": "Krill Size (mm/ Gr)"}, inplace=True)

        return df_to2018

    def extract_data_after_102018(client):
        """
        Extract data from MySQL from after October 2018; this data is in two tables that hence must be merged;
            - `akbm_vessels_dev_productionReporting.BridgeReporting`     (position data, starting December 2018)
            - `akbm_vessels_dev_productionReporting.FactoryReporting`    (catch volume data, starting December 2018)
        """
        query_2018onwards_bridge = """
                SELECT *
                FROM akbm_vessels_dev_productionReporting.BridgeReporting
            """
        query_2018onwards_factory = """
                SELECT *
                FROM akbm_vessels_dev_productionReporting.FactoryReporting
            """
        query_2018onwards_laboratory = """
                SELECT *
                FROM akbm_vessels_dev_productionReporting.LabReporting
            """
        df_2018_bridge = client.query(query_2018onwards_bridge).to_dataframe()
        df_2018_factory = client.query(query_2018onwards_factory).to_dataframe()
        df_2018_lab = client.query(query_2018onwards_laboratory).to_dataframe()

        # Change date columns to datetime
        df_2018_bridge['Date'] = pd.to_datetime(df_2018_bridge['Date'])
        df_2018_factory['Date'] = pd.to_datetime(df_2018_factory['Date'])
        df_2018_lab['Date'] = pd.to_datetime(df_2018_lab['Date'])

        # Group data from CatchFactory on vessel and date, as there are multiple rows with same vessel and date
        #     - one row per product type; Aqua 25 kg and Aqua 500 kg
        df_2018_factory_grouped = df_2018_factory.groupby(['Vessel', 'Date']).agg(
            {'RawKrillCatchKg': 'sum', 'Kg': 'sum'}).reset_index()

        # Group data from LabReporting on vessel and date, as there are multiple rows with same vessel and date
        df_2018_lab_grouped = df_2018_lab.groupby(['Vessel', 'Date']).agg(
            {'MM': 'mean', 'Gram': 'mean'}).reset_index()

        # Left merge on 'vessel' and 'date', keeping rows in df_2018_bridge without volume (no-catch days)
        df_from_102018 = pd.merge(df_2018_bridge, df_2018_factory_grouped, on=['Vessel', 'Date'], how='left')
        df_from_102018 = pd.merge(df_from_102018, df_2018_lab_grouped, on=['Vessel', 'Date'], how='outer')

        # Drop some columns
        df_from_102018.drop(['RowLoadedToCloudUTCDT'], axis=1, inplace=True)

        # Change volume from [kg] to [Mt]
        df_from_102018["RawKrillCatchKg"] = df_from_102018["RawKrillCatchKg"] * 0.001

        # Rename some columns to be consistent with dataset from before Dec 2018
        df_from_102018.rename(index=str, columns={"RawKrillCatchKg": "Total catch Krill - Mt",
                                                  "Kg": "Total Krill Meal Kg",
                                                  "WindDirection": "Wind direction",
                                                  "WaterTempC": "Water temp (Celsius)",
                                                  "BaricPressureHPa": "Baric pressure (hPa)",
                                                  "TrawlDepthM": "Trawl depth (m)",
                                                  "WindSpeedKm": "Wind speed (kn)",
                                                  "MM": "Krill Size (mm)",
                                                  "Gram": "Krill weight (gram)"}, inplace=True)
        return df_from_102018

    tic = time.time()
    cwd = os.getcwd()

    # 0) Setup connection to database in BigQuery on GCP
    if verbose == 1: print('Establishing connection to BigQuery in GCP.')
    client = BigQueryClient(config_path, config_dict=config_dict)
    try:
        _ = client.client.list_datasets()
        if verbose == 1: print('    SUCCESS: Connection to BigQuery in GCP established.')
    except (BadRequest) as e:
        print("CONNECTION FAILED: Unable to connect to BigQuery in GCP.")
        return


    # 1) First extract data to Oct.2018
    if verbose == 1: print('Extracting catch data from 2010 to October 2018.')
    df_to2018 = extract_data_to_102018(client.client)

    # 2) Then extract data from Dec.2018 onwards - position data in prod.CatchBridge and volume in prod.CatchFactory
    if verbose == 1: print('Extracting catch data from December 2018 onwards.')
    df_from_102018 = extract_data_after_102018(client.client)

    # 3) Then merge the two together
    if verbose == 1: print('Merging and cleaning catch data')
    df_catch = df_to2018.append(df_from_102018, ignore_index=True, sort=False)

    # 4) Run cleaning scripts and pre-processing, e.g. extracting lat-lon from position
    df_catch = clean_catch_data(df_catch)

    # 5) Save data to csv
    if savefile is True:
        enddate = str(max(df_catch['Date'])).replace('-', '')[:8]
        if verbose == 1: print("Saving data to '%s/data/All_catch_data_to_%s.csv'" % (cwd, enddate))
        df_catch.to_csv(cwd + 'data/All_catch_data_to_%s.csv' % enddate, index=False)

    if verbose == 1: print('DONE after %.2fsec: Data extracted, merged and cleaned.' % (time.time()-tic))

    return df_catch
