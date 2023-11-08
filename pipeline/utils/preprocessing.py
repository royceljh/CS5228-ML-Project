import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
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
    # town = row['town']
    # monthly_rent_by_town = town_mean_dict[town]
    rent_approval_year = row['rent_approval_year']
    rent_approval_month = int(row['rent_approval_month']) - 1
    
    year_month_monthly_coe_mean_ = year_month_monthly_coe_mean[rent_approval_year]
    year_month_monthly_coe_mean_value = year_month_monthly_coe_mean_[rent_approval_month]
    
    return year_month_monthly_coe_mean_value

def custom_func_quota(row):
    # town = row['town']
    # monthly_rent_by_town = town_mean_dict[town]
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
    # town = row['town']
    # monthly_rent_by_town = town_mean_dict[town]
    rent_approval_year = row['rent_approval_year']
    rent_approval_month = int(row['rent_approval_month']) - 1
    
    year_month_monthly_adjusted_close_mean_ = year_month_open[rent_approval_year]
    year_month_monthly_adjusted_close_mean_value = year_month_monthly_adjusted_close_mean_[rent_approval_month]
    
    return year_month_monthly_adjusted_close_mean_value

def custom_func_high(row):
    # town = row['town']
    # monthly_rent_by_town = town_mean_dict[town]
    rent_approval_year = row['rent_approval_year']
    rent_approval_month = int(row['rent_approval_month']) - 1
    
    year_month_monthly_adjusted_close_mean_ = year_month_high[rent_approval_year]
    year_month_monthly_adjusted_close_mean_value = year_month_monthly_adjusted_close_mean_[rent_approval_month]
    
    return year_month_monthly_adjusted_close_mean_value

def custom_func_low(row):
    # town = row['town']
    # monthly_rent_by_town = town_mean_dict[town]
    rent_approval_year = row['rent_approval_year']
    rent_approval_month = int(row['rent_approval_month']) - 1
    
    year_month_monthly_adjusted_close_mean_ = year_month_low[rent_approval_year]
    year_month_monthly_adjusted_close_mean_value = year_month_monthly_adjusted_close_mean_[rent_approval_month]
    
    return year_month_monthly_adjusted_close_mean_value

def custom_func_close(row):
    # town = row['town']
    # monthly_rent_by_town = town_mean_dict[town]
    rent_approval_year = row['rent_approval_year']
    rent_approval_month = int(row['rent_approval_month']) - 1
    
    year_month_monthly_adjusted_close_mean_ = year_month_close[rent_approval_year]
    year_month_monthly_adjusted_close_mean_value = year_month_monthly_adjusted_close_mean_[rent_approval_month]
    
    return year_month_monthly_adjusted_close_mean_value

def custom_func_adjusted_close(row):
    # town = row['town']
    # monthly_rent_by_town = town_mean_dict[town]
    rent_approval_year = row['rent_approval_year']
    rent_approval_month = int(row['rent_approval_month']) - 1
    
    year_month_monthly_adjusted_close_mean_ = year_month_adjusted_close[rent_approval_year]
    year_month_monthly_adjusted_close_mean_value = year_month_monthly_adjusted_close_mean_[rent_approval_month]
    
    return year_month_monthly_adjusted_close_mean_value

    
def generate_folds(df):    
    df['rent_approval_year'] = df['rent_approval_date'].apply(lambda x: x.split('-')[0])
    df['rent_approval_month'] = df['rent_approval_date'].apply(lambda x: x.split('-')[1])
    df = preprocess_geographic_location_pang(df)
    
    n_folds = 5
    df = create_folds(df, num_splits=n_folds)
    
    return df
    
def clean_replace_numeric(df, feature, search, replace):
    df[feature] = np.where(df[feature].str.contains(search, na=False), replace, df[feature])
    
def generate_dummies(df, feature):
    return pd.get_dummies(df[feature], feature)

def preprocess(df=None):
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