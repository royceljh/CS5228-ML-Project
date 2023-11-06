import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from math import radians, sin, cos, sqrt, atan2



def get_min(df, feature):
    print("Min: " + str(df[feature].min()))


def get_max(df, feature):
    print("Max: " + str(df[feature].max()))


def get_histogram(df, feature):
    plt.hist(df[feature], bins=10)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title("Histogram of " + str(feature))
    plt.show()


def get_density_plot(df, feature):
    sns.kdeplot(df[feature], shade=True)
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.title("Density Plot of " + str(feature))
    plt.show()


def get_counts(df, feature):
    value_counts = df[feature].value_counts().sort_values()
    print(value_counts)


def search_and_replace(df, feature, search, replace):
    return np.where(
        df[feature].str.contains(search, na=False), replace, df[feature]
    )


def format_numeric(df, feature):
    df[feature] = pd.to_numeric(df[feature], errors="coerce")


def plot_trend(df, feature):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=feature, y="monthly_rent")
    plt.title("Trend between " + str(feature) + " and monthly_rent")
    plt.show()


def plot_scatter(df, feature1, feature2):
    sns.scatterplot(data=df, x=feature1, y="monthly_rent", hue=feature2)
    plt.title("Rental price by " + str(feature1) + " and " + str(feature2))
    plt.xlabel(feature1)
    plt.ylabel("Monthly Rent")
    plt.show()


def generate_dummies(df, feature):
    return pd.get_dummies(df[feature], feature, drop_first=True)


def create_folds(data, num_splits):
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1

    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    num_bins = int(np.floor(1 + np.log2(len(data))))

    # bin targets
    data.loc[:, "bins"] = pd.cut(data["monthly_rent"], bins=num_bins, labels=False)
    kf = StratifiedKFold(n_splits=num_splits)

    # Using bins to split
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, "kfold"] = f

    data = data.drop("bins", axis=1)
    return data

def calculate_distance(lat1, lon1, other_df, year=None):
    r = 6371
    _other_df=other_df.copy()
    if year:
        _other_df = _other_df[_other_df['opening_year']<= year]
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat_diff = np.absolute(_other_df['latitude'] - lat1)
    lon_diff =np.absolute(_other_df['longitude'] - lon1)
    a = np.sin(lat_diff/2)**2 + np.cos(lat1) * np.cos(lon_diff) * np.sin(lon_diff / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return np.nanmin(c) * r

def add_stock_price_trend(df):
    stock_price = pd.read_csv('../auxiliary-data/sg-stock-prices.csv')
    # constituents of STI.
    STI = ['D05', 'O39', 'U11', 'Z74', 'J36', 'J37', 'H78', 'C09', 'C38U', 'BN4', 'F34', 'A17U', 'Y92', 'G13', 'C6L', 'V03', 'C52', 'ME8U', 'S68', 'N2IU']
    # we interested in stocks that are to STI only
    stock_price['symbol'] = stock_price['symbol'].apply(lambda x: x.replace('.SI', ''))
    stock_price = stock_price[stock_price['symbol'].str.contains('|'.join(STI))]
    # we look at price of the stock at closing
    stock_price = stock_price.drop(columns=['open','high','low','adjusted_close','symbol'])
    stock_price=stock_price.pivot(index='date',columns='name', values='close')
    # rename index for ease of joining later
    stock_price.index.rename('rent_approval_date',inplace=True)
    # convert index to datetime
    stock_price.index = pd.to_datetime(stock_price.index)
    stock_price_monthly_mean = stock_price.resample('M').mean()
    # scale it by avg closing price of first month
    stock_price_monthly_mean = stock_price_monthly_mean/stock_price_monthly_mean.iloc[0]
    stock_price_monthly_mean.index = stock_price_monthly_mean.index.strftime('%Y-%m')
    return df.join(stock_price_monthly_mean,on='rent_approval_date')

