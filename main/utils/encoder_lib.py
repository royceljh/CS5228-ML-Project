import category_encoders as ce
import pandas as pd

class TargetEncoderWrapper:
    
    def __init__(self, feature_col, target_col, smoothing):
        self.target_col = target_col
        self.feature_col = feature_col
        self.encoder = ce.TargetEncoder(smoothing=smoothing)

    # Fit and transform training dataset
    def fit_transform(self, df):
        X = df.drop(self.target_col, axis=1)
        y = df[self.target_col]
        X[self.feature_col] = self.encoder.fit_transform(X[self.feature_col], y)
        X[self.feature_col] = X[self.feature_col].round().astype("int")
        X = pd.concat([X, y], axis=1)
        return X

    # Transform testing dataset
    def transform(self, df):
        X = df.copy()
        X[self.feature_col] = self.encoder.transform(X[self.feature_col])
        X[self.feature_col] = X[self.feature_col].round().astype("int")
        return X
