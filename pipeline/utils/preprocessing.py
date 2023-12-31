import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pandas import DataFrame
from math import radians, sin, cos, sqrt, atan2
import sys
sys.path.append('../../')
import os
from .const import *
    
def format_numeric(df, feature):
    df[feature] = pd.to_numeric(df[feature], errors='coerce')

def format_integer(df, feature):
    df[feature] = df[feature].astype('int')

def create_folds(data, num_splits):
    """
        stratified k-fold split by binning the monthly_rent into equal distribution
        data: dataframe
        num_splits: number of folds to create
        return: dataframe with kfold column
    """
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    num_bins = int(np.floor(1 + np.log2(len(data))))
    
    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["monthly_rent"], bins=num_bins, labels=False
    )    
    kf = StratifiedKFold(n_splits=num_splits)
    
    # Using bins to split
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    
    data = data.drop("bins", axis=1)
    return data

def calculate_distance_pang(lat, lon, other_df):
    lat_diff = other_df['latitude'] - lat
    lon_diff = other_df['longitude'] - lon
    distances = np.sqrt(lat_diff**2 + lon_diff**2)
    return distances.min()


def preprocess_geographic_location(df:DataFrame) -> DataFrame :
    # code by Ding Ming
    # generates features: shortest distance to mrt, shortest distance to plan mrt, /
    # shortest distance to primary school, shortest distance to shopping mall, /
    # any primary school within 1km radius, any top primary school within 1 km radius

    def calculate_distance_using_lat_lon(lat1, lon1, other_df, year=None):
        # calculate distance using lat and long degrees between 2 points
        r = 6371
        _other_df=other_df.copy()
        if year:
            # year of rental input is required when calculating shortest distance to MRT station
            _other_df = _other_df[_other_df['opening_year']<= year]
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat_diff = np.absolute(_other_df['latitude'] - lat1)
        lon_diff =np.absolute(_other_df['longitude'] - lon1)
        a = np.sin(lat_diff/2)**2 + np.cos(lat1) * np.cos(lon_diff) * np.sin(lon_diff / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return np.nanmin(c) * r

    _df =df.copy()
    mrt_exist_df = pd.read_csv('../auxiliary-data/sg-mrt-existing-stations.csv')
    mrt_exist_df['latitude'] = mrt_exist_df['latitude'].apply(lambda x:radians(x))
    mrt_exist_df['longitude'] = mrt_exist_df['longitude'].apply(lambda x:radians(x))
    mrt_planned_df = pd.read_csv('../auxiliary-data/sg-mrt-planned-stations.csv')
    mrt_planned_df['latitude'] = mrt_planned_df['latitude'].apply(lambda x:radians(x))
    mrt_planned_df['longitude'] = mrt_planned_df['longitude'].apply(lambda x:radians(x))
    primary_school_df = pd.read_csv('../auxiliary-data/sg-primary-schools.csv')    
    primary_school_df['latitude'] = primary_school_df['latitude'].apply(lambda x:radians(x))
    primary_school_df['longitude'] = primary_school_df['longitude'].apply(lambda x:radians(x))
    top_primary_school = ['Rosyth School', 'Nan Hua Primary School','St. Hilda’s Primary School', 'Catholic High School', 'Henry Park Primary School','Nanyang Primary School','Tao Nan School','Anglo-Chinese School', 'Raffles Girls’ Primary School']
    top_primary_school_df = primary_school_df[primary_school_df['name'].isin(top_primary_school)]
    shopping_mall_df = pd.read_csv('../auxiliary-data/sg-shopping-malls.csv')    
    shopping_mall_df['latitude'] = shopping_mall_df['latitude'].apply(lambda x:radians(x))
    shopping_mall_df['longitude'] = shopping_mall_df['longitude'].apply(lambda x:radians(x))
    # Calculate distances for each row in df
    _df['dist_mrt_exist'] = _df.apply(lambda row: calculate_distance_using_lat_lon(row['latitude'], row['longitude'], mrt_exist_df,row['rent_approval_year']), axis=1)
    _df['dist_mrt_planned'] = _df.apply(lambda row: calculate_distance_using_lat_lon(row['latitude'], row['longitude'], mrt_planned_df), axis=1)
    _df['dist_primary_school'] = _df.apply(lambda row: calculate_distance_using_lat_lon(row['latitude'], row['longitude'], primary_school_df), axis=1)
    _df['within_1km_primary_school'] = _df['dist_primary_school'] < 1
    _df['dist_top_primary_school'] = _df.apply(lambda row: calculate_distance_using_lat_lon(row['latitude'], row['longitude'], top_primary_school_df), axis=1)
    _df['within_1km_top_primary_school'] = _df['dist_top_primary_school'] < 1
    _df['dist_shopping_mall'] = _df.apply(lambda row: calculate_distance_using_lat_lon(row['latitude'], row['longitude'], shopping_mall_df), axis=1)
    return _df

def preprocess_geographic_location_pang(df):
    _df =df.copy()
    mrt_exist_df = pd.read_csv('../../auxiliary-data/sg-mrt-existing-stations.csv')
    mrt_planned_df = pd.read_csv('../../auxiliary-data/sg-mrt-planned-stations.csv')
    primary_school_df = pd.read_csv('../../auxiliary-data/sg-primary-schools.csv')
    shopping_malls_df = pd.read_csv('../../auxiliary-data/sg-shopping-malls.csv')
    # Calculate distances for each row in df
    _df['dist_mrt_exist_pang'] = _df.apply(lambda row: calculate_distance_pang(row['latitude'], row['longitude'], mrt_exist_df), axis=1)
    _df['dist_mrt_planned_pang'] = _df.apply(lambda row: calculate_distance_pang(row['latitude'], row['longitude'], mrt_planned_df), axis=1)
    _df['dist_primary_school_pang'] = _df.apply(lambda row: calculate_distance_pang(row['latitude'], row['longitude'], primary_school_df), axis=1)
    _df['dist_shopping_malls_pang'] = _df.apply(lambda row: calculate_distance_pang(row['latitude'], row['longitude'], shopping_malls_df), axis=1)    
    return _df    
    
    
def custom_func(row):
    rent_approval_year = row['rent_approval_year']
    rent_approval_month = int(row['rent_approval_month']) - 1
    
    year_month_monthly_coe_mean_ = year_month_monthly_coe_mean[rent_approval_year]
    year_month_monthly_coe_mean_value = year_month_monthly_coe_mean_[rent_approval_month]
    
    return year_month_monthly_coe_mean_value

def custom_func_quota(row):
    rent_approval_year = row['rent_approval_year']
    rent_approval_month = int(row['rent_approval_month']) - 1
    
    year_month_monthly_coe_mean_ = year_month_monthly_quota_mean[rent_approval_year]
    year_month_monthly_coe_mean_value = year_month_monthly_coe_mean_[rent_approval_month]
    
    return year_month_monthly_coe_mean_value

def custom_func_bids(row):
    # town = row['town']
    # monthly_rent_by_town = town_mean_dict[town]
    rent_approval_year = row['rent_approval_year']
    rent_approval_month = int(row['rent_approval_month']) - 1
    
    year_month_monthly_coe_mean_ = year_month_monthly_bids_mean[rent_approval_year]
    year_month_monthly_coe_mean_value = year_month_monthly_coe_mean_[rent_approval_month]
    
    return year_month_monthly_coe_mean_value


def custom_func_open(row):
    rent_approval_year = row['rent_approval_year']
    rent_approval_month = int(row['rent_approval_month']) - 1
    
    year_month_monthly_adjusted_close_mean_ = year_month_open[rent_approval_year]
    year_month_monthly_adjusted_close_mean_value = year_month_monthly_adjusted_close_mean_[rent_approval_month]
    
    return year_month_monthly_adjusted_close_mean_value

def custom_func_high(row):
    rent_approval_year = row['rent_approval_year']
    rent_approval_month = int(row['rent_approval_month']) - 1
    
    year_month_monthly_adjusted_close_mean_ = year_month_high[rent_approval_year]
    year_month_monthly_adjusted_close_mean_value = year_month_monthly_adjusted_close_mean_[rent_approval_month]
    
    return year_month_monthly_adjusted_close_mean_value

def custom_func_low(row):
    rent_approval_year = row['rent_approval_year']
    rent_approval_month = int(row['rent_approval_month']) - 1
    
    year_month_monthly_adjusted_close_mean_ = year_month_low[rent_approval_year]
    year_month_monthly_adjusted_close_mean_value = year_month_monthly_adjusted_close_mean_[rent_approval_month]
    
    return year_month_monthly_adjusted_close_mean_value

def custom_func_close(row):
    rent_approval_year = row['rent_approval_year']
    rent_approval_month = int(row['rent_approval_month']) - 1
    
    year_month_monthly_adjusted_close_mean_ = year_month_close[rent_approval_year]
    year_month_monthly_adjusted_close_mean_value = year_month_monthly_adjusted_close_mean_[rent_approval_month]
    
    return year_month_monthly_adjusted_close_mean_value

def custom_func_adjusted_close(row):
    rent_approval_year = row['rent_approval_year']
    rent_approval_month = int(row['rent_approval_month']) - 1
    
    year_month_monthly_adjusted_close_mean_ = year_month_adjusted_close[rent_approval_year]
    year_month_monthly_adjusted_close_mean_value = year_month_monthly_adjusted_close_mean_[rent_approval_month]
    
    return year_month_monthly_adjusted_close_mean_value

    
def generate_folds(df):    
    df['rent_approval_year'] = df['rent_approval_date'].apply(lambda x: x.split('-')[0])
    df['rent_approval_month'] = df['rent_approval_date'].apply(lambda x: x.split('-')[1])
    df = preprocess_geographic_location_pang(df)
    
    n_folds = 25
    df = create_folds(df, num_splits=n_folds)
    
    return df
    
def clean_replace_numeric(df, feature, search, replace):
    df[feature] = np.where(df[feature].str.contains(search, na=False), replace, df[feature])
    
def generate_dummies(df, feature):
    return pd.get_dummies(df[feature], feature)

def add_stock_price_trend(df):
    # code by Ding Ming
    # This function enhance the train dataset by incorporating information about stock prices.
    # generates features: normalized stock price of the STI constituents.
    # joint on rent_approval_date
    # Due to its poor perforamance, we did not use these features in the final model
    
    # Load stock price data
    stock_price = pd.read_csv('../auxiliary-data/sg-stock-prices.csv')
    
    # List of constituents of STI.
    STI = ['D05', 'O39', 'U11', 'Z74', 'J36', 'J37', 'H78', 'C09', 'C38U', 'BN4', 'F34', 'A17U', 'Y92', 'G13', 'C6L', 'V03', 'C52', 'ME8U', 'S68', 'N2IU']
    
    # Remove '.SI' from symbol names and filter for STI stocks only
    stock_price['symbol'] = stock_price['symbol'].apply(lambda x: x.replace('.SI', ''))
    stock_price = stock_price[stock_price['symbol'].str.contains('|'.join(STI))]
    
    # Keep only closing prices
    stock_price = stock_price.drop(columns=['open', 'high', 'low', 'adjusted_close', 'symbol'])
    
    # Pivot the data to have dates as index and stock names as columns
    stock_price = stock_price.pivot(index='date', columns='name', values='close')
    
    # Rename index for ease of joining later
    stock_price.index.rename('rent_approval_date', inplace=True)
    
    # Convert index to datetime
    stock_price.index = pd.to_datetime(stock_price.index)
    
    # Resample the data to get monthly mean prices
    stock_price_monthly_mean = stock_price.resample('M').mean()
    
    # Scale the data by dividing by the average closing price of the first month
    stock_price_monthly_mean = stock_price_monthly_mean / stock_price_monthly_mean.iloc[0]
    
    # Format index to 'yyyy-mm'
    stock_price_monthly_mean.index = stock_price_monthly_mean.index.strftime('%Y-%m')
    
    # Join the stock price data with the original DataFrame on 'rent_approval_date'
    return df.join(stock_price_monthly_mean, on='rent_approval_date')

def preprocess(df=None):
    """
        preprocess train dataframe to generate features
        df: train dataframe
        return: preprocessed dataframe with 285 features
    """        
    df = pd.read_csv('../../data/train.csv')
    sg_coe_df = pd.read_csv('../../auxiliary-data/sg-coe-prices.csv')
    sg_coe_df['month'] = sg_coe_df['month'].apply(lambda x: month_to_int[x])
    df = generate_folds(df)

    clean_replace_numeric(df, 'flat_type', '2', 2)
    clean_replace_numeric(df, 'flat_type', '3', 3)
    clean_replace_numeric(df, 'flat_type', '4', 4)
    clean_replace_numeric(df, 'flat_type', '5', 5)
    clean_replace_numeric(df, 'flat_type', 'executive', 6)

    town_dummies = generate_dummies(df, 'town')
    flat_model_dummies = generate_dummies(df, 'flat_model')
    # train['flat_type'] = train['flat_type'].astype(int)
    flat_type_dummies = generate_dummies(df, 'flat_type')
    lease_commence_dummies = generate_dummies(df, 'lease_commence_date')
    region_dummies = generate_dummies(df, 'region')
    rent_approval_year_dummies = generate_dummies(df, 'rent_approval_year')
    rent_approval_month_dummies = generate_dummies(df, 'rent_approval_month')
    subzone_dummies = generate_dummies(df, 'subzone')
    planning_area_dummies = generate_dummies(df, 'planning_area')
    
    
    df['flat_type'] = df['flat_type'].astype(int)
    
    df['year_month_monthly_coe_mean'] = df.apply(custom_func, axis=1)
    df['year_month_monthly_quota_mean'] = df.apply(custom_func_quota, axis=1)
    df['year_month_monthly_bids_mean'] = df.apply(custom_func_bids, axis=1)
    
    
    sg_stock = pd.read_csv('../../auxiliary-data/sg-stock-prices.csv')
    sg_stock['year'] = sg_stock['date'].apply(lambda x: x.split('-')[0]).astype(str)
    sg_stock['month'] = sg_stock['date'].apply(lambda x: x.split('-')[1]).astype(int)
    
    
    df['year_month_adjusted_close'] = df.apply(custom_func_adjusted_close, axis=1)
    df['year_month_open'] = df.apply(custom_func_open, axis=1)
    df['year_month_high'] = df.apply(custom_func_high, axis=1)
    df['year_month_low'] = df.apply(custom_func_low, axis=1)
    df['year_month_close'] = df.apply(custom_func_close, axis=1)
    
    train = pd.concat([
                region_dummies, town_dummies, flat_model_dummies, flat_type_dummies, lease_commence_dummies,
                rent_approval_year_dummies, rent_approval_month_dummies,
                subzone_dummies,
               df[['floor_area_sqm',                   
                   'dist_mrt_exist_pang', 'dist_mrt_planned_pang', 'dist_primary_school_pang', 'dist_shopping_malls_pang',
                   'year_month_monthly_coe_mean', 'year_month_monthly_quota_mean', 'year_month_monthly_bids_mean',
                   'year_month_adjusted_close', # 'year_month_open', 'year_month_high', 'year_month_low', 'year_month_close',
                   'kfold', 'monthly_rent']],
          ], axis=1)
    
    return train
