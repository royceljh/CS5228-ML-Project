import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold


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


def clean_replace_numeric(df, feature, search, replace):
    df[feature] = np.where(
        df[feature].str.contains(search, na=False), replace, df[feature]
    )


def format_numeric(df, feature):
    df[feature] = pd.to_numeric(df[feature], errors="coerce")


def format_integer(df, feature):
    df[feature] = df[feature].astype("int")


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


def target_encode(df, feature):
    target_encoder_smooth = ce.TargetEncoder(cols=[feature], smoothing=10)
    target_encoder_smooth.fit(df[feature], df["monthly_rent"])
    df[feature] = target_encoder_smooth.transform(df[feature])
    df[feature] = df[feature].round()


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
