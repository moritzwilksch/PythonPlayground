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


# for catcol in categoricals:
#     X_train_original[catcol] = X_train_original[catcol].astype("category")
#     X_train_original[catcol] = X_train_original[catcol].cat.add_categories(["missing"])
#     X_train_original[catcol] = X_train_original[catcol].fillna("missing")
#     X_train_original[catcol] = X_train_original[catcol].cat.codes
X_train_original[categoricals] = X_train_original[categoricals].astype("category")


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
        .str.strptime(pl.Date, "%Y-%B-%d")
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

#%%
