# træn logistisk regression på skadedata

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from feature_engine.selection import (
    DropConstantFeatures,
    DropCorrelatedFeatures,
)
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from workshop.utils import DropFeatures

# indlæs "rådata" for 2015-18
df = pd.read_parquet("data/train.parquet")

# beregn binomialt target, der angiver, om der har været skade(r) eller ej
y = df["skadesantal"].values > 0

# instantiér generator til split af data, 'random_state' sættes for at kunne reproducere split
split_generator = ShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

# generér indices for de to datasæt
train_index, val_index = next(split_generator.split(df))

# opdel data i trænings- og valideringssæt
X_train, y_train = df.iloc[train_index], y[train_index]
X_val, y_val = df.iloc[val_index], y[val_index]

# drop arbitrære features, definér blacklist til senere brug
blacklist = [
    # 'skadesudgift' og 'skadesantal' er direkte afledt af target-variablen, og kan derfor ikke bruges til at prædiktere den
    "skadesudgift",
    "skadesantal",
    # vi fjerner 'aar', da vi indtil videre bortser fra eventuelle tidslige mønstre i data
    "aar",
    # 'idnummer' er en unik identifikator for hver kunde, der ikke - i sig selv - indeholder signal
    "idnummer",
]

# Feature-transformationer
## Numeriske features
transformer_num = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

## Kategoriske features
transformer_cat = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
        ),
    ]
)

## Sæt sammen til ColumnTransformer
col_transformer = ColumnTransformer(
    transformers=[
        (
            "transformer_num",
            transformer_num,
            make_column_selector(dtype_include=["number"]),
        ),
        (
            "transformer_cat",
            transformer_cat,
            make_column_selector(dtype_include=["category"]),
        ),
    ],
    verbose_feature_names_out=False,
)

## Komplet pipeline
pipe = Pipeline(
    [
        ("drop_features_blacklist", DropFeatures(blacklist)),
        ("drop_constant_features", DropConstantFeatures(missing_values="ignore")),
        ("col_transformer", col_transformer),
        (
            "drop_correlated_features",
            DropCorrelatedFeatures(threshold=0.95, missing_values="ignore"),
        ),
        ("predictor", LogisticRegression(max_iter=1000, solver="newton-cholesky")),
    ]
)

# Model tuning
## Definér parameter-grid, der skal gennemsøges
param_grid = {
    "predictor__C": [1e-5, 1],
    "col_transformer__transformer_num__imputer__strategy": ["mean", "median"],
}

# Forbered GridSearch
grid = GridSearchCV(
    pipe,
    param_grid,
    cv=split_generator,
    scoring="roc_auc",
    return_train_score=True,
    n_jobs=-1,
    verbose=4,
)

# Udfør grid search
print("Træner model(ler)")
grid.fit(df, y)

print("Disse parametre giver den bedste performance: ", grid.best_params_)
print("Med en ROC AUC på: ", grid.best_score_)

# Udskriv resultater
results = pd.DataFrame.from_records(grid.cv_results_["params"])
results["mean_valid_score"] = grid.cv_results_["mean_test_score"]
results["mean_train_score"] = grid.cv_results_["mean_train_score"]
print("Resultater for grid search:")
print(results.sort_values("mean_valid_score", ascending=False))

# SUBMIT LØSNING

# import pandas as pd
# from workshop.utils import submit
# # indlæs test-data
# test_set = pd.read_parquet("data/test.parquet")
# # beregn prædikterede sandsynligheder på test-sættet
# y_score = grid.predict_proba(test_set)[:, 1]
# # sandsynlighederne skal være en `list`
# y_score = y_score.tolist()
# # herefter sendes løsningen med `submit`
# submit(y_score)
