import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class DataPreprocess:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean_column_names(self):
        self.df.columns = [col.strip().lower().replace(" ", "_") for col in self.df.columns]
        return self

    def fill_missing(self, strategy="mean", fill_value=None):
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if strategy == "mean" and self.df[col].dtype != "O":
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == "median" and self.df[col].dtype != "O":
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == "mode":
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == "constant":
                    self.df[col].fillna(fill_value, inplace=True)
        return self

    def encode_column(self, col, mapping: dict):
        if col in self.df.columns:
            self.df[col] = self.df[col].map(mapping)
        else:
            raise KeyError(f"column {col} does not exist")
        return self

    def encode_nominal(self, col):
        if col in self.df.columns:
            dummies = pd.get_dummies(self.df[col], prefix=col).astype(int)
            self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
        else:
            raise KeyError(f"column {col} does not exist")
        return self

    def class_process(self, col):
        self.df[col] = self.df.pop(col)
        self.df[col + "_num"] = self.df[col].map({"Yes": 1, "No": 0})

    def min_max_scaler(self, columns):
        for col in columns:
            scaler = MinMaxScaler()
            self.df[[col]] = scaler.fit_transform(self.df[[col]])
        return self

    def get_data(self):
        return self.df

    def save_csv(self, filepath):
        self.df.to_csv(filepath, index=False)
        return self