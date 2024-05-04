import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import requests
import pandas as pd
import numpy as np

_freq_col_types = {
    "idnummer": "string",
    "aar": "Int64",
    "Kat": "category",
    "expo": "float",
    "skadesantal": "Int64",
    "alder": "Int64",
    "forsum": "Int64",
    "zone": "category",
    "anc": "Int64",
    "afst_brand": "float",
    "afst_politi": "float",
    "anthund": "Int64",
    "boligareal": "float",
    "selvrisk": "Int64",
    "segment": "category",
    "ant_vaadrum": "Int64",
    "betfrekvens": "category",
    "byg_anvend_kode": "category",
    "kon": "category",
    "lejlighed": "category",
    "opfaar": "Int64",
    "parcel": "category",
    "tagtype": "category",
    "varmeinst": "category",
    "geo": "category",
}


def load_freq_data(file="freq.csv"):
    df = pd.read_csv("freq.csv", dtype=_freq_col_types)
    return df


_claims_col_types = {
    "idnummer": "string",
    "aar": "Int64",
    "Kat": "category",
    "skadesudgift": "float",
}


def load_claims_data(file="claims.csv"):
    df = pd.read_csv("claims.csv", dtype=_claims_col_types)
    return df


def load_data():
    freq = load_freq_data()
    claims = load_claims_data()

    # summér skadesudgifter på idnummer, kat og aar i claims
    claims = (
        claims.groupby(["idnummer", "Kat", "aar"], observed=True)["skadesudgift"]
        .sum()
        .reset_index(name="skadesudgift")
    )

    # merge skader og skadesudgifter
    freq = freq.merge(claims, how="left", on=["idnummer", "Kat", "aar"])

    # håndtér missing values i skadesudgifter
    freq["skadesudgift"] = freq["skadesudgift"].fillna(0)

    return freq


class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_remove):
        self.columns_to_remove = columns_to_remove

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so we simply return self.
        return self

    def transform(self, X):
        # Return a new array with the specified columns removed.
        # We use numpy indexing to select the columns to keep.
        return X.drop(columns=self.columns_to_remove, axis=1)


class RemoveCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop = [
            column for column in upper.columns if any(upper[column] > self.threshold)
        ]
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        df = df.drop(df.columns[self.to_drop], axis=1)
        return df.values


def load_skade_data(data_path="data/skader.parquet"):
    df = pd.read_parquet(data_path)

    df = df[df["Kat"] == "Indbrud"]
    df = df[df["expo"] == 1]
    df = df[df["aar"] < 2019]

    # create binomial target, y
    y = df["skadesantal"].values > 0

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=0
    )

    return X_train, X_test, y_train, y_test


def _generate_datasets():
    df = load_data()
    df = df[df["expo"] == 1]
    df = df[df["Kat"] == "Indbrud"]

    test = df[df["aar"] == 2019]
    train = df[df["aar"] < 2019]

    y_test = test.skadesantal > 0
    y_test = y_test.to_numpy()

    # slør target-derivater
    test["skadesantal"] = 0
    test["skadesudgift"] = 0

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train.to_parquet("train.parquet")
    test.to_parquet("test.parquet")
    np.save("y_test.npy", y_test)


def submit(y_score: list, domain="united-lark-literate"):
    assert isinstance(
        y_score, list
    ), "'y_score' skal være en 'list' med sandsynligheder"
    assert len(y_score) == 153908, "'y_score' skal indeholde 153.908 sandsynligheder"

    url = f"https://{domain}.ngrok-free.app/submit"
    data = {"user": os.environ["GITHUB_USER"], "y_score": y_score}
    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("Din løsning er registreret! \N{grinning face} ")
    else:
        raise Exception(
            "Din løsning blev ikke registreret. Sandsynligvis fordi serveren, der skal behandle den, har travlt. Prøv at submitte igen."
        )
