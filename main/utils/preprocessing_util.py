import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce

def get_min(df, feature):
    print('Min: ' + str(df[feature].min()))
    
def get_max(df, feature):
    print('Max: ' + str(df[feature].max()))

def get_histogram(df, feature):
    plt.hist(df[feature], bins=10)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title('Histogram of ' + str(feature))
    plt.show()

def get_density_plot(df, feature):
    sns.kdeplot(df[feature], shade=True)
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.title('Density Plot of ' + str(feature))
    plt.show()

def get_counts(df, feature):
    value_counts = df[feature].value_counts().sort_values()
    print(value_counts)

def clean_replace_numeric(df, feature, search, replace):
    df[feature] = np.where(df[feature].str.contains(search, na=False), replace, df[feature])

def format_numeric(df, feature):
    df[feature] = pd.to_numeric(df[feature], errors='coerce')

def plot_trend(df, feature):
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x=feature, y='monthly_rent')
    plt.title('Trend between ' + str(feature) + ' and monthly_rent')
    plt.show()

def plot_scatter(df, feature1, feature2):
    sns.scatterplot(data=df, x=feature1, y='monthly_rent', hue=feature2)
    plt.title('Rental price by ' + str(feature1) + ' and ' + str(feature2))
    plt.xlabel(feature1)
    plt.ylabel('Monthly Rent')
    plt.show()

def target_encode(df, feature):
    target_encoder_smooth = ce.TargetEncoder(cols=[feature], smoothing=10)
    target_encoder_smooth.fit(df[feature], df['monthly_rent'])
    df[feature] = target_encoder_smooth.transform(df[feature])
    df[feature] = df[feature].round()
