# træn XGBoost på case-data

from sklearn.model_selection import ShuffleSplit, GridSearchCV
from xgboost import XGBClassifier, plot_importance
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from feature_engine.selection import (
    DropConstantFeatures,
)
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from workshop.utils import DropFeatures

# indlæs "rådata" for 2015-18
df = pd.read_parquet("data/train.parquet")

# dan binomial "target"-variabel, der angiver, om der har været skade(r) eller ej
y = df["skadesantal"].values > 0

# instantiér generator til split af data, random state sættes for at kunne reproducere split
split_generator = ShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

# generér indices for de to splits
train_index, val_index = next(split_generator.split(df))

# opdel data i trænings- og valideringssæt
X_train, y_train = df.iloc[train_index], y[train_index]
X_val, y_val = df.iloc[val_index], y[val_index]

## Pipeline
blacklist = [
    "aar",
    "idnummer",
    "skadesudgift",
    "skadesantal",
]

pipe = Pipeline(
    [
        ("drop_features_blacklist", DropFeatures(blacklist)),
        ("drop_constant_features", DropConstantFeatures(missing_values="ignore")),
        ("predictor", XGBClassifier(
            objective="binary:logistic",
            enable_categorical=True,
        ))
    ]
)

## Modeltuning
# Definér parameter-grid, der skal gennemsøges
param_grid = {
    "predictor__learning_rate": [0.1, 0.3],
    "predictor__max_depth": [3, 6],
    "predictor__n_estimators": [100, 250],
}

# Forbered GridSearch
# Bemærk, at vi her bruger vores split_generator til at generere trænings- og valideringssæt
grid = GridSearchCV(
    pipe,
    param_grid,
    cv=split_generator,
    scoring="roc_auc",
    return_train_score=True,
    n_jobs=-1,
    verbose=4,
)

# Udfør grid search - bemærk, vi giver hele datasættet (2015-18) som input og overlader det til split-generatoren at splitte det i trænings- og valideringssæt
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
