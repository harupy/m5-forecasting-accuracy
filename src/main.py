# %% [markdown]
# This kernel is:
# - Based on [Very fst Model](https://www.kaggle.com/ragnar123/very-fst-model). Thanks @ragnar123!
# - Automatically uploaded by [push-kaggle-kernel](https://github.com/harupy/push-kaggle-kernel).
# - Formatted by [Black](https://github.com/psf/black).

# %% [markdown]
# # Objective

# * Make a baseline model that predict the validation (28 days).
# * This competition has 2 stages, so the main objective is to make a model that can predict the demand for the next 28 days.

# %% [code]
import gc
import os
import warnings

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)

# %% [code]
import IPython


def display(*dfs, head=True):
    for df in dfs:
        IPython.display.display(df.head() if head else df)


# %% [code]
def on_kaggle():
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


# %% [code]
if on_kaggle():
    os.system("pip install --quiet mlflow_extend")


# %% [code]
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


# %% [code]
def read_data():
    INPUT_DIR = "/kaggle/input" if on_kaggle() else "input"
    INPUT_DIR = f"{INPUT_DIR}/m5-forecasting-accuracy"

    print("Reading files...")

    calendar = pd.read_csv(f"{INPUT_DIR}/calendar.csv").pipe(reduce_mem_usage)
    sell_prices = pd.read_csv(f"{INPUT_DIR}/sell_prices.csv").pipe(reduce_mem_usage)
    sales_train_val = pd.read_csv(f"{INPUT_DIR}/sales_train_validation.csv").pipe(
        reduce_mem_usage
    )
    submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv").pipe(
        reduce_mem_usage
    )

    print("Calendar shape:", calendar.shape)
    print("Sell prices shape:", sell_prices.shape)
    print("Sales train shape:", sales_train_val.shape)
    print("Submission shape:", submission.shape)

    return calendar, sell_prices, sales_train_val, submission


# %% [code]
calendar, sell_prices, sales_train_val, submission = read_data()


# %% [code]
def melt(
    sales_train_val, submission, nrows=55_000_000, verbose=True,
):
    # melt sales data, get it ready for training
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales_train_val = pd.melt(
        sales_train_val, id_vars=id_columns, var_name="day", value_name="demand",
    )

    sales_train_val = reduce_mem_usage(sales_train_val)

    if verbose:
        display(sales_train_val)

    # separate test dataframes.
    test1 = submission[submission["id"].str.contains("validation")]
    test2 = submission[submission["id"].str.contains("evaluation")]

    if verbose:
        display(test1, test2)

    # change column names.
    test1.columns = ["id"] + [f"d_{x}".format(x) for x in range(1914, 1914 + 28)]
    test2.columns = ["id"] + [f"d_{x}".format(x) for x in range(1942, 1942 + 28)]

    # get product table.
    product = sales_train_val[id_columns].drop_duplicates()

    # merge with product table
    test2["id"] = test2["id"].str.replace("_evaluation", "_validation")
    test1 = test1.merge(product, how="left", on="id")
    test2 = test2.merge(product, how="left", on="id")
    test2["id"] = test2["id"].str.replace("_validation", "_evaluation")

    if verbose:
        display(test1, test2)

    test1 = pd.melt(test1, id_vars=id_columns, var_name="day", value_name="demand")
    test2 = pd.melt(test2, id_vars=id_columns, var_name="day", value_name="demand")

    if verbose:
        display(test1, test2)

    sales_train_val["part"] = "train"
    test1["part"] = "test1"
    test2["part"] = "test2"

    data = pd.concat([sales_train_val, test1, test2], axis=0)

    if verbose:
        display(data)

    del sales_train_val, test1, test2

    # get only a sample for fast training.
    data = data.loc[nrows:]

    # delete test2 for now.
    data = data[data["part"] != "test2"]

    gc.collect()

    return data


def merge(data, calendar, sell_prices, verbose=True):
    # drop some calendar features.
    calendar = calendar.drop(["weekday", "wday", "month", "year"], axis=1)

    # notebook crashes with the entire dataset.
    data = pd.merge(data, calendar, how="left", left_on=["day"], right_on=["d"])
    data = data.drop(["d", "day"], axis=1)

    if verbose:
        display(data)

    # get the sell price data (this feature should be very important).
    data = data.merge(sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    if verbose:
        display(data)

    return data


# %% [code]
data = melt(sales_train_val, submission, nrows=27_500_000)
data = merge(data, calendar, sell_prices)
data = reduce_mem_usage(data)


# %% [code]
def encode_categoricals(df):
    nan_features = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    for feature in nan_features:
        df[feature] = df[feature].fillna("MISSING")

    cat_cols = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]

    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

    return df


def add_agg_features(df):
    # rolling demand features.
    for shift in [28, 29, 30]:
        df[f"shift_t{shift}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(shift)
        )

    for size in [7, 30]:
        df[f"rolling_std_t{size}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(28).rolling(size).std()
        )

    for size in [7, 30, 90, 180]:
        df[f"rolling_mean_t{size}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(28).rolling(size).mean()
        )

    df["rolling_skew_t30"] = df.groupby(["id"])["demand"].transform(
        lambda x: x.shift(28).rolling(30).skew()
    )
    df["rolling_kurt_t30"] = df.groupby(["id"])["demand"].transform(
        lambda x: x.shift(28).rolling(30).kurt()
    )

    # price features
    df["shift_price_t1"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.shift(1)
    )
    df["price_change_t1"] = (df["shift_price_t1"] - df["sell_price"]) / (
        df["shift_price_t1"]
    )
    df["rolling_price_max_t365"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.shift(1).rolling(365).max()
    )
    df["price_change_t365"] = (df["rolling_price_max_t365"] - df["sell_price"]) / (
        df["rolling_price_max_t365"]
    )

    df["rolling_price_std_t7"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.rolling(7).std()
    )
    df["rolling_price_std_t30"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.rolling(30).std()
    )
    return df.drop(["rolling_price_max_t365", "shift_price_t1"], axis=1)


def add_time_features(df):
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.week
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6])
    return df


# %% [code]
data = encode_categoricals(data)
data = add_agg_features(data)
data = add_time_features(data)
data = reduce_mem_usage(data)


# %% [code]
def train_lgb(bst_params, fit_params, X, y, cv):
    models = []

    for idx_fold, (idx_trn, idx_val) in enumerate(cv.split(X, y)):
        print(f"\n---------- Fold: ({idx_fold + 1} / {cv.get_n_splits()}) ----------\n")

        X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
        y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]
        train_set = lgb.Dataset(X_trn, label=y_trn)
        val_set = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            bst_params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            **fit_params,
        )
        models.append(model)

    return models


# %% [code]
features = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "sell_price",
    # aggregation features.
    "shift_t28",
    "shift_t29",
    "shift_t30",
    "rolling_mean_t7",
    "rolling_std_t7",
    "rolling_mean_t30",
    "rolling_mean_t90",
    "rolling_mean_t180",
    "rolling_std_t30",
    "price_change_t1",
    "price_change_t365",
    "rolling_price_std_t7",
    "rolling_price_std_t30",
    "rolling_skew_t30",
    "rolling_kurt_t30",
    # time features.
    "year",
    "month",
    "week",
    "day",
    "dayofweek",
]

# prepare training and test data.
# 2011-01-29 ~ 2016-04-24 : d_1    ~ d_1913
# 2016-04-25 ~ 2016-05-22 : d_1914 ~ d_1941 (public)
# 2016-05-23 ~ 2016-06-19 : d_1942 ~ d_1969 (private)

mask1 = data["date"] <= "2016-03-27"  # noqa
mask2 = data["date"] <= "2016-04-24"  # noqa
X_train = data[mask2][features]
y_train = data[mask2]["demand"]

# X_val = data[~mask1 & mask2][features]
# y_val = data[~mask1 & mask2]["demand"]

X_test = data[~mask2][features]
id_date = data[~mask2][["id", "date"]]  # keep these two columns to use later.

del data
gc.collect()

# %% [code]
bst_params = {
    "boosting_type": "gbdt",
    "metric": "rmse",
    "objective": "regression",
    "n_jobs": -1,
    "seed": 42,
    "learning_rate": 0.1,
    "bagging_fraction": 0.75,
    "bagging_freq": 10,
    "colsample_bytree": 0.75,
}

fit_params = {
    "num_boost_round": 100_000,
    "early_stopping_rounds": 50,
    "verbose_eval": 100,
}

cv = TimeSeriesSplit(n_splits=5)
models = train_lgb(bst_params, fit_params, X_train, y_train, cv)


# %% [code]
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# %% [code]
imp_type = "gain"
importances = np.zeros(X_train.shape[1])
preds = np.zeros(X_test.shape[0])

for model in models:
    preds += model.predict(X_test)
    importances += model.feature_importance(imp_type)

# Take the average over folds.
preds = preds / cv.get_n_splits()
importances = importances / cv.get_n_splits()

# %% [markdown]
# https://github.com/harupy/mlflow-extend

# %% [code]
from mlflow_extend import mlflow, plotting as mplt

with mlflow.start_run():
    mlflow.log_params_flatten(
        {
            "bst": bst_params,
            "fit": fit_params,
            "cv": {"type": str(TimeSeriesSplit), "n_splits": cv.get_n_splits()},
        }
    )


features = models[0].feature_name()
_ = mplt.feature_importance(features, importances, imp_type, limit=30)


# %% [code]
def make_submission(test, submission):
    preds = test[["id", "date", "demand"]]
    preds = pd.pivot(preds, index="id", columns="date", values="demand").reset_index()
    F_cols = ["F" + str(x + 1) for x in range(28)]
    preds.columns = ["id"] + F_cols

    evals = submission[submission["id"].str.contains("evaluation")]
    vals = submission[["id"]].merge(preds, how="inner", on="id")
    final = pd.concat([vals, evals])

    assert final[F_cols].isnull().sum().sum() == 0

    final.to_csv("submission.csv", index=False)


# %% [code]
make_submission(id_date.assign(demand=preds), submission)
