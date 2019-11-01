"""
Pre-processing and feature engineering of AKBM catch data, in particular on the aggregated dataset.
"""
import numpy as np
import pandas as pd
from utils.geofunctions_utils import get_latitude_longitude_as_dd, get_moon_phase, distance_between_latlon_coords


def clean_catch_data(df_catch):
    """
    Clean aggregated dataset.
    Note there are lots of hard-coded rules in this function, e.g. column names.
    Parameters
    ----------
        df_catch -- pandas DataFrame, raw data
    Returns
    -------
        df_catch -- pandas DataFrame, cleaned data
    """
    # Make two new column; [latitude, longitude] which are taken from position column (in N W format)
    df_catch['Position'] = df_catch['Position'].apply(lambda x: correct_position(x))
    df_catch.dropna(subset=['Position'], inplace=True)   # Remove rows with missing position
    df_catch.reset_index(inplace=True, drop=True)
    df_catch['latlon_tuple'] = df_catch['Position'].apply(lambda x: get_latitude_longitude_as_dd(x))
    df_catch[['Latitude', 'Longitude']] = df_catch['latlon_tuple'].apply(pd.Series)
    df_catch.drop(['latlon_tuple'], axis=1, inplace=True)

    # Change production day column to datetime
    for col in ['Date', 'Production day']:
        if col in df_catch.columns:
            df_catch[col] = pd.to_datetime(df_catch[col])

    # Change selected column types from object (strings) to numeric floats
    dict_replace_chars = {'-': '', ' ': '', ',': '.', 'nan': '', '%': '', 'ND': ''}

    cols_to_float = ['Total catch Krill - Mt', 'Total Krill Meal Kg', 'Yield %',
                     'Krill Size (mm)', 'Krill weight (gram)']
    for col in cols_to_float:
        if col in df_catch.columns:
            for org, replacement in dict_replace_chars.items():
                df_catch[col] = df_catch[col].apply(lambda x: str(x).replace(org, replacement))
            df_catch[col] = pd.to_numeric(df_catch[col])

    # Extract krill weight and length from combined string column if it exists
    if 'Krill Size (mm/ Gr)' in df_catch.columns:
        temp_df_size = df_catch['Krill Size (mm/ Gr)'].apply(lambda x: split_krill_size(x)).apply(pd.Series)
        temp_df_size.rename(index=str, columns={0: "Krill Size (mm)", 1: "Krill weight (gram)"}, inplace=True)
        for col in temp_df_size.columns:
            temp_df_size[col] = temp_df_size[col].astype('float64')

        # Combine the two
        array_length_1 = temp_df_size['Krill Size (mm)']
        array_length_2 = df_catch['Krill Size (mm)']
        array_weight_1 = temp_df_size['Krill weight (gram)']
        array_weight_2 = df_catch['Krill weight (gram)']
        array_combined = np.zeros([len(array_length_1), 2])

        for i in range(len(array_length_1)):
            # Set krill length to the one that is valid
            if array_length_1[i] > 0:
                array_combined[i, 0] = array_length_1[i]
            elif array_length_2[i] > 0:
                array_combined[i, 0] = array_length_2[i]
            else:
                array_combined[i, 0] = None

            # Set krill weight to the one that is valid
            if array_weight_1[i] > 0:
                array_combined[i, 1] = array_weight_1[i]
            elif array_weight_2[i] > 0:
                array_combined[i, 1] = array_weight_2[i]
            else:
                array_combined[i, 1] = None
        df_catch[['Krill Size (mm)', 'Krill weight (gram)']] = array_combined
        df_catch.drop(['Krill Size (mm/ Gr)'], axis=1, inplace=True)

    # Extract wind direction and speed from combined column if it exists
    if 'Wind' in df_catch.columns:
        temp_df_wind = df_catch['Wind'].apply(lambda x: split_wind_string(x)).apply(pd.Series)
        temp_df_wind.rename(index=str, columns={0: "Wind direction", 1: "Wind speed (kn)"}, inplace=True)

        # Combine the two
        array_winddir_1 = temp_df_wind['Wind direction'].values
        array_winddir_2 = df_catch['Wind direction'].values
        winddirections = []
        array_windspeed_1 = temp_df_wind['Wind speed (kn)'].values
        array_windspeed_2 = df_catch['Wind speed (kn)'].values
        windspeeds = [] #np.zeros([len(array_winddir_1), 2])

        for i in range(len(array_winddir_1)):
            # Set wind direction to the one that is valid
            if pd.isnull(array_winddir_1[i]) is False:
                winddir = array_winddir_1[i]
            elif pd.isnull(array_winddir_2[i]) is False:
                winddir = array_winddir_2[i]
            else:
                winddir = None
            winddirections.append(winddir)

            # Set krill weight to the one that is valid
            if pd.isnull(array_windspeed_1[i]) is False:
                windspeed = array_windspeed_1[i]
            elif pd.isnull(array_windspeed_2[i]) is False:
                windspeed = array_windspeed_2[i]
            else:
                windspeed = None
            windspeeds.append(windspeed)
        df_catch['Wind direction'] = winddirections
        df_catch['Wind speed (kn)'] = windspeeds
        df_catch.drop(['Wind'], axis=1, inplace=True)

    # Finally order by date
    df_catch.sort_values(by=["Date"], inplace=True)
    df_catch.reset_index(inplace=True, drop=True)

    return df_catch


def feature_engineering_catch_data(df_catch):
    """
    Feature engineering on cleaned AKBM catch dataset, in particular extract the following features:
        - moon phase
        - Days since last report
        - daily travel distance ("distance since yesterday")

    More features should be naturally be added, this will be the goal for Q2 2019.

    Parameters
    ----------
        df_catch -- pandas DataFrame, cleaned data
    Returns
    -------
        df_catch_Antarctic -- pandas DataFrame, data for Antarctic Sea
        df_catch_Saga -- pandas DataFrame, data for Saga Sea
    """
    # Make individual datasets per vessel and sort by date
    df_catch_Antarctic = df_catch[df_catch['Vessel'] == 'Antarctic Sea'].copy()
    df_catch_Saga = df_catch[df_catch['Vessel'] == 'Saga Sea'].copy()
    df_catch_Endurance = df_catch[df_catch['Vessel'] == 'Antarctic Endurance'].copy()

    # Sort by date
    df_catch_Antarctic.sort_values(by=['Date'], inplace=True)
    df_catch_Saga.sort_values(by=['Date'], inplace=True)
    if df_catch_Endurance.empty is False:
        df_catch_Endurance.sort_values(by=['Date'], inplace=True)

    # NOT USED currently - skipping - give date on dd/mm/yyyy to fix
    # Compute moon phase for given day
    # for temp_df in [df_catch_Antarctic, df_catch_Saga, df_catch_Endurance]:
    #     if temp_df.empty is False:
    #         temp_df['Moon phase'] = temp_df['Date'].apply(lambda x: get_moon_phase(x.year, x.month, x.day))

    # Compute days since last catch/report day
    for temp_df in [df_catch_Antarctic, df_catch_Saga, df_catch_Endurance]:
        if temp_df.empty is False:
            days_diff = temp_df['Date'] - temp_df['Date'].shift()
            temp_df['Days since last report'] = days_diff.map(lambda x: x.days)

    # Calculate distance travelled since yesterday
    for temp_df in [df_catch_Antarctic, df_catch_Saga, df_catch_Endurance]:
        if temp_df.empty is False:
            n = temp_df.shape[0]
            latlons = temp_df[['Latitude', 'Longitude']].values
            distances = np.zeros([n, 1])
            for i in np.arange(1, n):
                if temp_df['Days since last report'].iloc[i] == 1:
                    dist = distance_between_latlon_coords(latlons[i, 0], latlons[i, 1], latlons[i - 1, 0],
                                                          latlons[i - 1, 1])
                    distances[i] = dist
            temp_df['Distance since yesterday'] = distances

    return df_catch_Antarctic, df_catch_Saga, df_catch_Endurance


def clean_catch_data_williamdata(df_catch):
    """
    Clean aggregated dataset.
    Note there are lots of hard-coded rules in this function, e.g. column names.

    Parameters
    ----------
        df_catch -- pandas DataFrame, raw data

    Returns
    -------
        df_catch -- pandas DataFrame, cleaned data
    """
    # Change column headers where unnamed
    df_catch = df_catch.rename(columns={'Unnamed: 8': 'Comments',
                                        'Unnamed: 46': 'Comments_raw_krill',
                                        'Unnamed: 52': 'Comments_onboard_analysis'})

    # Make two new column; [latitude, longitude] whcih are taken from position column (in N W format)
    #df_catch['Position'].replace('-', '', inplace=True)
    #df_catch['Position'].fillna('', inplace=True)
    df_catch['Position'] = df_catch['Position'].apply(lambda x: correct_position(x))
    #latlon = pd.DataFrame(columns=['S_W', 'S', 'W', 'Latitude', 'Longitude'])
    #latlon['S_W'] = df_catch['Position'].copy()

    #for i in range(latlon.shape[0]):
    #    el = latlon['S_W'].iloc[i].split()
    #    if len(el) == 4:
    #        latlon['S'].iloc[i] = el[0]
    #        latlon['W'].iloc[i] = el[2]
    #        temp_lat, temp_lon = get_latitude_longitude_as_dd(latlon['S_W'].iloc[i])
    #        latlon['Latitude'].iloc[i] = temp_lat
    #        latlon['Longitude'].iloc[i] = temp_lon
    #df_catch['Latitude'] = pd.to_numeric(latlon['Latitude'].fillna(''))
    #df_catch['Longitude'] = pd.to_numeric(latlon['Longitude'].fillna(''))
    # Convert position to lat-lon
    df_catch['latlon_tuple'] = df_catch['Position'].apply(lambda x: get_latitude_longitude_as_dd(x))
    df_catch[['Latitude', 'Longitude']] = df_catch['latlon_tuple'].apply(pd.Series)
    df_catch.drop(['latlon_tuple'], axis=1, inplace=True)

    # Change selected column types from object (strings) to numeric floats
    dict_replace_chars = {'-': '', ' ': '', ',': '.', 'nan': '', '%': '', 'ND': ''}

    cols_to_float = ['Total catch Krill (Mt)', 'Total Krill Meal (Kg)', 'Yield (%)',
                     'Krill Size (mm)', 'Krill Size (g)', 'Water Activity_QAKM',
                     'Moisture%_QAKM', 'Protein%_QAKM', 'Fat%_QAKM',
                     'Ash%_QAKM', 'Salt%_QAKM', 'Fluorine%_QAKM', 'FEQ 500 usage (l)_QAKM',
                     'FEQ 500 estimated ppm_QAKM', 'EQ measured ppm_QAKM',
                     'QRILL Astaxanthin Oil  Coulometri',
                     'Total Saturated _QAKM',
                     'C22:6, n-3 (dha)_QAKM',
                     'Total PUFA Omega-3_QAKM',
                     'Ratio PUFA/SAFA_QAKM',
                     'Total Saturated _Raw krill',
                     'C20:4, n-3_Raw krill',
                     'C21:5, n-3_Raw krill',
                     'C22:5, n-3 (dpa)_Raw krill',
                     'C22:6, n-3 (dha)_Raw krill',
                     'Total PUFA Omega-3_Raw krill',
                     'Ratio PUFA/SAFA_Raw krill']
    for col in cols_to_float:
        for org, replacement in dict_replace_chars.items():
            df_catch[col] = df_catch[col].apply(lambda x: str(x).replace(org, replacement))
        df_catch[col] = pd.to_numeric(df_catch[col])

    # Will drop disease/bacteria analysis columns for now
    cols_to_drop = ['Coliform cfu/g', 'E.coli cfu/g', 'Enterob. cfu/g',
                    'Aerobic cfu/g', 'Salmonella', 'Comments_onboard_analysis']
    df_catch.drop(cols_to_drop, axis=1, inplace=True)

    return df_catch


def correct_position(pos_string):
    """
    Correct position string, should be on the format (XXºXXS XXºXXW).
    Also replaces invalid positions with None (e.g. '' and '00º00S 00º00W')
    Returns on format (XXºXX S XXºXX W)

    Some misspellings encountered:
        [XX°XXSXXXX°W]
        [xx°xxS xx°xxW], i.e. characters 'xx' in position
        [XX°XX'XX''S XX°XX'XX''W], e.g. [63°12'50''S 58°11'50''W] on 09 June 2018
    """
    if pos_string is not None:
        ### NOTE: Should add a check here to ensure position is not reversed, i.e. XXºXXW XXºXXS instead XXºXXS XXºXXW

        ###
        for char in ['S', 'W', 'N', 'E', '°', 'º', '.', ',', "'", '-']:
            pos_string = str(pos_string).replace(char, ' ')
        el_list = pos_string.split()

        # Check that the numbers in 'el_list' are of length 2; if of length 4 then split in 2
        updated_list = []
        for el in el_list:
            if len(el) == 4:
                updated_list.append(el[:2])
                updated_list.append(el[2:])
            elif len(el) == 2:
                updated_list.append(el)
        el_list = updated_list

        if 'xx' in el_list:  # some occurrences of xx°xxS xx°xxW
            pos_string = None
            return pos_string

        if len(el_list) < 4:
            pos_string = None
            return pos_string

        if len(el_list) == 6:
            el_list = [el_list[0], el_list[1], el_list[3], el_list[4]]

        # Check that all elements in list are valid (should be integers with two digits, or numbers < 90)
        for i in range(len(el_list)):
            if el_list[i][0] == '0' and len(el_list[i]) > 1:
                el_list[i] = el_list[i][1:]
            if len(el_list[i]) > 2:
                el_list[i] = el_list[i][:2]

        # Assure minutes are valid - should be between 0 and 60
        if int(el_list[1]) >= 60:
            el_list[0] = str(int(el_list[0]) + 1)
            el_list[1] = str(int(el_list[1]) - 60)
        if int(el_list[3]) >= 60:
            el_list[2] = str(int(el_list[2]) + 1)
            el_list[3] = str(int(el_list[3]) - 60)

        latitude = "%02d" % (int(el_list[0])) + '°' + "%02d" % (int(el_list[1])) + ' S '
        longitude = "%02d" % (int(el_list[2])) + '°' + "%02d" % (int(el_list[3])) + ' W'
        pos_string = latitude + longitude

        # Check if position is valid
        if el_list[0] == '00' or el_list[0] == '00':
            pos_string = None
    return pos_string


def split_krill_size(size_string):
    """
    Correct krill size, i.e. split up string 'XX.Xmm/Y.YYg' into tuple with krill length and weight.
    Returns on format (XX.X, Y.YY)
    """
    size_string = str(size_string).lower()

    # Change ',' to '.'
    size_string = size_string.replace(',', '.')
    size_string = size_string.replace('..', '.')

    # Remove potential spaces between numbers and '.'
    size_string = size_string.replace(' ', '')

    # Replace characters and / with spaces and then split on spaces
    for char in ['m', '/', 'g', 'r', 'k', 'n', 'a', '-']:
        size_string = size_string.replace(char, ' ')
    el_list = size_string.split()

    if len(el_list) == 2:
        length = el_list[0]
        weight = el_list[1]
        # return length, weight
        return float(length), float(weight)
    else:
        return None, None


def split_wind_string(wind_string):
    """
    Correct and split manually entered wind specifications
    Returns a tuple (wind_direction, wind_speed) with types (str, float)
    NOTE: There is a lot of customization in this function based on manually cleaning the data.
          May therefor not generalize to future manually entered wind data
    """
    if wind_string == '' or pd.isnull(wind_string):
        return None, None
    else:
        wind_string = str(wind_string)

        # Replace ',' with '.'
        wind_string = wind_string.replace(',', '.')

        # Replace specific wind directions
        repl = (('ESE', 'SE'), ('WSE', 'SE'), ('WSW', 'SW'), ('ESW', 'SW'), ('SSE', 'SE'), ('SSW', 'SW'),
                ('E-SE', 'SE'), ('W-SE', 'SE'), ('W-SW', 'SW'), ('E-SW', 'SW'), ('S-SE', 'SE'), ('S-SW', 'SW'),
                ('ENE', 'NE'), ('WNE', 'NE'), ('WNW', 'NW'), ('ENW', 'NW'), ('NNE', 'NE'), ('NNW', 'NW'),
                ('E-NE', 'NE'), ('W-NE', 'NE'), ('W-NW', 'NW'), ('E-NW', 'NW'), ('N-NE', 'NE'), ('N-NW', 'NW'))
        for r in repl:
            wind_string = wind_string.replace(*r)

        repl = (('East', ' E '), ('West', ' W '), ('North', ' N '), ('South', ' S '),
                ('Nice and calm', 'Calm'), ('calm', 'Calm'))
        for r in repl:
            wind_string = wind_string.replace(*r)

        # Replace characters with spaces
        for char in ['-', 'wind', 'kts', 'knots', 'Knots', 'kots', 'kn']:
            wind_string = wind_string.replace(char, ' ')

        el_list = wind_string.split()

        if len(el_list) == 2:
            direction = el_list[0]
            strength = el_list[1]
            # return length, weight
            return direction, float(strength)
        elif len(el_list) == 1:
            try:
                strength = float(el_list[0])
                return None, strength
            except ValueError:
                # If value can not be interpreted as number, we choose it to be the direction (a string)
                direction = el_list[0]
                return direction, None
