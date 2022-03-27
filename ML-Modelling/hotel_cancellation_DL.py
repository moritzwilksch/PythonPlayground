#%%
import pandas as pd
import numpy as np
import pickle

################################
# Load the data
################################
data = pickle.load(open("a5_q1.pkl", "rb"))

y_train = data["y_train"]
X_train_original = data["X_train"]  # Original dataset
X_train_ohe = data["X_train_ohe"]  # One-hot-encoded dataset

X_test_original = data["X_test"]
X_test_ohe = data["X_test_ohe"]

################################
# Produce submission
################################


def create_submission(confidence_scores, save_path):
    """Creates an output file of submissions for Kaggle

    Parameters
    ----------
    confidence_scores : list or numpy array
        Confidence scores (from predict_proba methods from classifiers) or
        binary predictions (only recommended in cases when predict_proba is
        not available)
    save_path : string
        File path for where to save the submission file.

    Example:
    create_submission(my_confidence_scores, './data/submission.csv')

    """
    import pandas as pd

    submission = pd.DataFrame({"score": confidence_scores})
    submission.to_csv(save_path, index_label="id")


#%%
categoricals = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
]


X_train_original[categoricals] = X_train_original[categoricals].astype("category")
for catcol in categoricals:
    X_train_original[catcol] = X_train_original[catcol].cat.codes


#%%
import polars as pl

df = pl.from_pandas(X_train_original)
df

#%%
df = df.with_columns(
    [
        # weekday
        pl.concat_str(
            ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"],
            sep="-",
        )
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        .dt.weekday()
        .alias("arrival_weekday"),
        # total stay
        pl.sum(["stays_in_weekend_nights", "stays_in_week_nights"]).alias(
            "total_nights_stayed"
        ),
        # room changed
        (pl.col("assigned_room_type") != pl.col("reserved_room_type")).alias(
            "room_changed"
        ),
    ]
)

#%%
rare_agents = (
    df.select(pl.col("agent").value_counts())
    .unnest("agent")
    .filter(pl.col("counts") < 10)["agent"]
    .to_list()
)

df[df.select(pl.col("agent").is_in(rare_agents)).to_numpy(), "agent"] = 0

#%%
df = df.to_pandas()

#%%
from sklearn.model_selection import train_test_split

cols_to_drop = [
    "company",
    # "arrival_date_day_of_month",
    # "arrival_date_week_number",
    # "agent",
    "arrival_date_year",
]
xtrain, xval, ytrain, yval = train_test_split(
    df.drop(cols_to_drop, axis=1), y_train, test_size=0.2, random_state=42
)

# # LGBM max val AUC 0.88 (accuracy: 0.89)

# #%%
# from sklearn.impute import KNNImputer, SimpleImputer

# imputer = SimpleImputer()
# numcols = xtrain.select_dtypes(exclude="category").columns
# xtrain[numcols] = pd.DataFrame(imputer.fit_transform(xtrain[numcols]), columns=numcols)
# xval[numcols] = pd.DataFrame(imputer.fit_transform(xval[numcols]), columns=numcols)

# #%%
# from lightgbm import LGBMClassifier

# # clf = LGBMClassifier(n_estimators=10_000)
# # clf.fit(
# #     xtrain,
# #     ytrain,
# #     eval_metric=["auc", "binary_logloss"],
# #     eval_set=(xval, yval),
# #     # categorical_feature=categoricals,
# #     early_stopping_rounds=500,
# # )

# from sklearn.ensemble import RandomForestClassifier

# clf = RandomForestClassifier(n_jobs=-1, max_depth=10)
# clf.fit(xtrain, ytrain)

# #%%
# from sklearn.metrics import classification_report, roc_auc_score

# print(classification_report(ytrain, clf.predict(xtrain)))
# print(classification_report(yval, clf.predict(xval)))
# print(roc_auc_score(yval, clf.predict(xval)))

# #%%
# from imblearn.over_sampling import SMOTE, SMOTENC

# smote = SMOTENC(
#     n_jobs=-1, categorical_features=[c in categoricals for c in xtrain.columns]
# )
# xtrain_smote, ytrain_smote = smote.fit_resample(xtrain, ytrain)
# #%%
# import optuna


# def objective(trial: optuna.Trial):
#     params = {
#         "max_depth": trial.suggest_int("max_depth", 2, 100),
#         "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
#         "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
#         "max_features": trial.suggest_float("max_features", 0.1, 1),
#     }

#     clf = RandomForestClassifier(n_jobs=-1, **params)
#     clf.fit(xtrain_smote, ytrain_smote)
#     return roc_auc_score(yval, clf.predict(xval))


# study = optuna.create_study(
#     storage="sqlite:///optuna_study.db",
#     direction="maximize",
#     study_name="moritz_rf",
#     load_if_exists=True,
# )

# study.optimize(objective, n_trials=100)

#%%
import tensorflow as tf

catcols_embedding = {
    "hotel": 2,
    "arrival_date_month": 5,
    "meal": 2,
    "country": 10,
    "market_segment": 5,
    "distribution_channel": 2,
    "reserved_room_type": 2,
    "assigned_room_type": 2,
    "deposit_type": 2,
    "customer_type": 2,
}
catcols_cardinalities = df[categoricals].nunique().to_dict()


class HotelModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embeddings = {
            col: tf.keras.layers.Embedding(
                catcols_cardinalities.get(col) + 1, catcols_embedding.get(col)
            )
            for col in categoricals
        }
