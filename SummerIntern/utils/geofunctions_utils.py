"""
These are utils/helper functions for working with geospatial data, e.g. for coordinate convertions, moon phases etc. 
"""
from math import sin, cos, sqrt, atan2, radians
import numpy as np
import datetime
import ephem
from tempfile import TemporaryFile
import re
import math
import urllib.request
import json
# import sys 
# sys.path.append('/home/jupyter/catch/') # setting sys path to include project-root-folder
# from utils.get_moon_phase import *

def matrix_mask_from_seaicedata(seaice_data, data_source='NASA'):
    """
    Make matrix mask based on seaice data (numpy 2D array), i.e. a mask matrix giving where
    there is land (True) or not.

    Note that the returned mask matrix has 1 where we should mask, i.e. where there is land.
    This follows the convention of numpys mask module, see
        https://docs.scipy.org/doc/numpy/reference/maskedarray.generic.html

    Arguments:
    ----------
        seaice_data -- numpy npzfile object with fields ['seaice', 'latitude', 'longitude', 'date]
        data_source -- string, default 'NASA', should be in ['NASA', 'ESA'] (meereisportal uses ESA)
            NASA seaice data is categorical, where category 25 is land
            ESA seaice data is numeric (0 to 100), where nan is land
    """
    data_dim = len(seaice_data['seaice'].shape)
    if data_dim == 2:
        temp_data = seaice_data['seaice']
        #matrix_mask = ( == land_category)
    elif data_dim == 3:
        temp_data = seaice_data['seaice'][:, :, 0]
        #matrix_mask = ( == land_category)

    # Make mask matrix
    if data_source == 'NASA':
        matrix_mask = (temp_data == 25)
        #if data_dim == 2:
        #    matrix_mask = (seaice_data['seaice'] == land_category)
        #elif data_dim == 3:
        #    matrix_mask = (seaice_data['seaice'][:, :, 0] == land_category)
    elif data_source == 'ESA':
        matrix_mask = np.isnan(temp_data)
    matrix_mask = matrix_mask * 1

    #print(matrix_mask.shape)
    #print(np.sum(matrix_mask.ravel()))
    #print(matrix_mask)

    # Combine it with latitude and longitude in a numpy NpzFile
    outfile = TemporaryFile()
    np.savez_compressed(outfile, mask=matrix_mask,
             latitude=seaice_data['latitude'],
             longitude=seaice_data['longitude'])
    outfile.seek(0)
    mask_npz = np.load(outfile)

    return mask_npz


def get_location_temperature(lat, lon, date, data=None, datapath='../data/satellite_data/temperature.npz'):
    """
    Function to extract temperature data for specific location (lat, lon)
    for a certain date (or time period).

    Parameters
    ----------
        lat, lon -- floats
            Position in (latitude, longitude) for which we want to extract temperature
        date --
        data -- numpy NpzFile (numpy.lib.npyio.NpzFile) object, default None
            Temperature data file with fields ['temperature', 'latitude',
            'longitude', 'date']
        datapath -- string, default '../data/satellite_data/temperature.npz'
            Path to data file (if input object 'data' is None)

    Return
    ------
        location_temp -- float
            Temperature at specified/wanted location

    Notes
    -----
    - We assume here that the array "data['latitude']" has equal columns,
      "data['longitude']" has equal rows.

    """
    # Load data
    if data is None:
        data = np.load(datapath)
    temperature = data['temperature']   # Note - need to include filtering on date
    latitude = data['latitude']
    latitude_vector = latitude[0, :]
    longitude = data['longitude']
    longitude_vector = longitude[:, 0]

    # Find data for specified position
    index_row = np.argmin(np.abs(longitude_vector - lon))
    index_col = np.argmin(np.abs(latitude_vector - lat))

    location_temp = temperature[index_row, index_col, :]

    return location_temp


def get_latitude_longitude_as_dd(position):
    elements = position.split()
    ns_factor = [1.0, 1.0]  # factor to multiply with, differing between N (1), S (-1), E (1), W (-1)
    if len(elements) not in [2, 4]:
        raise Exception("Unexpected number of elements in position string '{pos}'"
                        .format(pos=position))
    elif len(elements) == 2:
        latitude = get_decimal_degrees(elements[0])
        longitude = get_decimal_degrees(elements[1])
    elif len(elements) == 4:
        latitude = get_decimal_degrees(elements[0])
        longitude = get_decimal_degrees(elements[2])

        if elements[1] == 'S':
            ns_factor[0] = -1.0
        if elements[3] == 'W':
            ns_factor[1] = -1.0

    try:
        latitude = latitude * ns_factor[0]
        longitude = longitude * ns_factor[1]

        return latitude, longitude
    except:
        return latitude, longitude


def get_decimal_degrees(value):
    """Get decimal degrees from string

    Expected formats: 'd.m', 'dºm', 'd°m' and 'd', where d is degrees and m
    is minutes (with a variable length of digits).
    """
    degrees_separators = ['.', 'º', '°']
    for separator in degrees_separators:
        if value.find(separator) != -1:
            parts = value.split(separator)
            if len(parts) != 2:
                raise Exception("Unexpected number of elements in value '{val}'"
                                .format(val=value))
            if parts[0].isdigit() and parts[1].isdigit():
                degrees = float(int(parts[0]))
                minutes = get_decimal_minutes(parts[1])

                return degrees + minutes / 60.0

    # try:
    #    return float(int(value))
    # except Exception:
    #    raise Exception("No separator matched input '{val}'".format(val=value))


def get_decimal_minutes(minutes_string):
    """Get decimal minutes from string

    Expects a value between 0 and 599... with a variable length of digits.

    Example input: '0', '59', '599999'.

    Args:
        minutes_string: String to be converted
    Returns:
        float: in the range [0,60)
    """
    minutes = int(minutes_string)
    number_of_digits = len(minutes_string)

    if number_of_digits > 1 and minutes_string[0] > '5':
        raise Exception("Leading minutes cannot be greater than 59, got {val} "
                        + "in first position", val=minutes_string[0])

    return minutes / 10 ** (number_of_digits - 2)


# import math

def deg_to_dms(deg, type='lat'):
    decimals, number = math.modf(deg)
    d = int(number)
    m = int(decimals * 60)
    s = (deg - d - m / 60) * 3600.00
    compass = {
        'lat': ('N','S'),
        'lon': ('E','W')
    }
    compass_str = compass[type][0 if d >= 0 else 1]
    return '{:2.0f}.{:2.0f}{}'.format(abs(d),abs(m), compass_str)

def convert_deg_to_dms(lat,lon):
    lat_string = deg_to_dms(lat,type='lat')
    lon_string = deg_to_dms(lon,type='lon')
    return "{},{}".format(lat_string,lon_string)


def distance_between_latlon_coords(lat1, lon1, lat2, lon2):
    """
    Compute distance in (km) between two coordinates.

    From https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude

    Parameters
    ----------
        lat1, lon1 -- floats, latitude and longitude for position 1
        lat2, lon2 -- floats, latitude and longitude for position 2

    Returns
    -------
        distance -- float, disctance between coordinates in (km)
    """
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance


def get_phase_on_day(year, month, day):
    """
    Returns a floating-point number from 0-1. where 0=new, 0.5=full, 1=new
    """
    # Ephem stores its date numbers as floating points, which the following uses
    # to conveniently extract the percent time between one new moon and the next
    # This corresponds (somewhat roughly) to the phase of the moon.

    # Use Year, Month, Day as arguments
    date = ephem.Date(datetime.date(year, month, day))

    nnm = ephem.next_new_moon(date)
    pnm = ephem.previous_new_moon(date)

    lunation = (date - pnm) / (nnm - pnm)

    # Note that there is a ephem.Moon().phase() command, but this returns the
    # percentage of the moon which is illuminated. This is not really what we want.
    # print(ephem.Moon().phase(date))
    return lunation


def get_moon_phase(date):
    """
    ephem.Moon().moon_phase command - returns the ratio of the moon which is illuminated.
    
    date a string on the format 'mm/dd/YYYY'
    """
    
    month = np.int(date[0:2])
    day = np.int(date[3:5])
    year = np.int(date[6:10])
    
    date = datetime.date(year, month, day)
    moon = ephem.Moon(date)

    return moon.moon_phase


errMsg = ""

def getUrl(date, coords):
    ApiURLBase = "https://api.usno.navy.mil/rstt/oneday?date=mDDYYYY&coords=60.00N,10.00E&tz=0"
    ApiURLBase = ApiURLBase.replace("mDDYYYY", date).replace("60.00N,10.00E", coords)
    return  ApiURLBase

def getData(url):
    ApiResp = ""
    global errMsg
    try:
        ApiResp = str(urllib.request.urlopen(url).read())
        ApiResp = ApiResp[2:-1].replace('\\n', '')
        
        if ApiResp.find("curphase")==-1:
            ApiResp = ApiResp[:-1] + ', "curphase":""}' # append curphase for the days it doesn't display so it can be later replaced with closestphase
        contentsDict = json.loads(ApiResp)

    finally:

        if contentsDict["error"]==False: # if no error returned by the API
            return contentsDict
        elif ApiResp == "": # if connection error
            errMsg = "Error while connecting to API: check if the Internet connection exists."
            return ""
        else:
            errMsg = "An error happened while requesting data from the API. Check the format of the input variables."  
            return ""


# def getMoonPhase(date, coords, info):

#     """
#     call getMoonPhase():
#     pass date in format "4/22/2019"
#     coords in format "59.54N,10.46E"
#     info: pass one of these
#             "moonphase" - phase name, returns string
#             "fracillum" - % of the Moon's lit surface, returns fraction
#     example:
#             getMoonPhase("4/23/2019", "50.05N,14.25E", "fracillum")

#     documentation of the API: https://aa.usno.navy.mil/data/docs/api.php
 
#     the script doesn't validate the input so be sure to follow the starting guidelines
#     """

#     ApiUrl = getUrl(date, coords)
#     dataString = getData(ApiUrl)

#     if info=="moonphase":
#         infoType = "curphase"
#     elif info == "fracillum":
#         infoType = "fracillum"
#     else:
#         infoType = "error"

#     if dataString == "":
#         return errMsg
#     elif infoType == "error":
#         return 'Desired datatype not specified. Choose "moonphase" or "fracillum" as the third argument.'
#     else: # success!
#         if dataString["curphase"]=="" and info == "moonphase": # (if "closestphase" occurs on date requested, "curphase" will not be in JSON)
#             return dataString["closestphase"]["phase"]
#         elif infoType == 'fracillum':
#             percent = int(dataString["fracillum"].replace("%", ""))
#             return percent/100
#         else:
#             return dataString[infoType]
        
def getMoonPhase(date, coords, info):

    """
    call getMoonPhase():
    pass date in format "4/22/2019"
    coords in format "59.54N,10.46E"
    info: pass one of these
            "moonphase" - phase name, returns string
            "fracillum" - % of the Moon's lit surface, returns fraction
    example:
            getMoonPhase("4/23/2019", "50.05N,14.25E", "fracillum")

    documentation of the API: https://aa.usno.navy.mil/data/docs/api.php
 
    the script doesn't validate the input so be sure to follow the starting guidelines
    """

    ApiUrl = getUrl(date, coords)
    dataString = getData(ApiUrl)

    if info=="moonphase":
        infoType = "curphase"
    elif info == "fracillum":
        infoType = "fracillum"
    else:
        infoType = "error"

    if dataString == "":
        return errMsg
    elif infoType == "error":
        return 'Desired datatype not specified. Choose "moonphase" or "fracillum" as the third argument.'
    else: # success!
        if dataString["curphase"]=="" and info == "moonphase": # (if "closestphase" occurs on date requested, "curphase" will not be in JSON)
            return dataString["closestphase"]["phase"]
        elif infoType == 'fracillum':
            try:
                percent = int(dataString["fracillum"].replace("%", ""))
            except:
                try:
                    phase = dataString["closestphase"]["phase"]
                    if phase == 'Full Moon':
                        percent = 100
                    elif phase == 'New Moon':
                        percent = 0
                    else:
                        print('Something is wrong - returning NaN')
                        percent = np.nan
                        print('Failing on {}'.format(date))
                except:
                    percent = np.nan
            return percent/100
        else:
            return dataString[infoType]

