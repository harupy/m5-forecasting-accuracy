# %% [markdown]
# This kernel is:
# - Based on [Very fst Model](https://www.kaggle.com/ragnar123/very-fst-model). Thanks @ragnar!
# - Automatically uploaded by [push-kaggle-kernel](https://github.com/harupy/push-kaggle-kernel).
# - Formatted by [Black](https://github.com/psf/black)

# %% [markdown]
# # Objective

# * Make a baseline model that predict the validation (28 days).
# * This competition has 2 stages, so the main objective is to make a model that can predict the demand for the next 28 days.

# %% [code]
import gc
import os
import sys
import warnings

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import preprocessing, metrics

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)


# %% [code]
def on_kaggle():
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


# %% [code]
if on_kaggle():
    os.system("git clone https://github.com/harupy/m5-forecasting-accuracy")
    os.system("pip install mlflow_extend")
    sys.path.append("m5-forecasting-accuracy/src")
else:
    sys.path.append("src")

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


# function to read the data and merge it (ignoring some columns, this is a very fst model)

# %% [code]
def read_data():
    input_dir = "/kaggle/input" if on_kaggle() else "input"
    input_dir = f"{input_dir}/m5-forecasting-accuracy"

    print("Reading files...")

    calendar = pd.read_csv(f"{input_dir}/calendar.csv").pipe(reduce_mem_usage)
    sell_prices = pd.read_csv(f"{input_dir}/sell_prices.csv").pipe(reduce_mem_usage)
    sales_train_val = pd.read_csv(f"{input_dir}/sales_train_validation.csv").pipe(
        reduce_mem_usage
    )
    submission = pd.read_csv(f"{input_dir}/sample_submission.csv").pipe(
        reduce_mem_usage
    )

    print("Calendar shape:", calendar.shape)
    print("Sell prices shape:", sell_prices.shape)
    print("Sales train shape:", sales_train_val.shape)
    print("Submission shape:", submission.shape)

    return calendar, sell_prices, sales_train_val, submission


def melt_and_merge(
    calendar, sell_prices, sales_train_val, submission, nrows=55_000_000, merge=False,
):
    # melt sales data, get it ready for training
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales_train_val = pd.melt(
        sales_train_val, id_vars=id_columns, var_name="day", value_name="demand",
    )
    sales_train_val = reduce_mem_usage(sales_train_val)

    # separate test dataframes
    test1 = submission[submission["id"].str.contains("validation")]
    test2 = submission[submission["id"].str.contains("evaluation")]

    # change column names
    test1.columns = ["id"] + [f"d_{x}".format(x) for x in range(1914, 1914 + 28)]
    test2.columns = ["id"] + [f"d_{x}".format(x) for x in range(1942, 1942 + 28)]

    # get product table
    product = sales_train_val[id_columns].drop_duplicates()

    # merge with product table
    test2["id"] = test2["id"].str.replace("_evaluation", "_validation")
    test1 = test1.merge(product, how="left", on="id")
    test2 = test2.merge(product, how="left", on="id")
    test2["id"] = test2["id"].str.replace("_validation", "_evaluation")

    test1 = pd.melt(test1, id_vars=id_columns, var_name="day", value_name="demand",)
    test2 = pd.melt(test2, id_vars=id_columns, var_name="day", value_name="demand",)

    sales_train_val["part"] = "train"
    test1["part"] = "test1"
    test2["part"] = "test2"

    data = pd.concat([sales_train_val, test1, test2], axis=0)

    del sales_train_val, test1, test2

    # get only a sample for fast training
    data = data.loc[nrows:]

    # drop some calendar features
    calendar = calendar.drop(["weekday", "wday", "month", "year"], axis=1)

    # delete test2 for now
    data = data[data["part"] != "test2"]

    if merge:
        # notebook crashes with the entire dataset
        data = pd.merge(data, calendar, how="left", left_on=["day"], right_on=["d"])
        data = data.drop(["d", "day"], axis=1)
        # get the sell price data (this feature should be very important)
        data = data.merge(
            sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
        )
    gc.collect()

    return data


# %% [code]
calendar, sell_prices, sales_train_val, submission = read_data()
data = melt_and_merge(
    calendar, sell_prices, sales_train_val, submission, nrows=27_500_000, merge=True
)

# %% [markdown]
# * We have the data to build our first model, let's build a baseline and predict the validation data (in our case is test1)

# %% [code]
def transform(data):
    nan_features = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    for feature in nan_features:
        data[feature] = data[feature].fillna("MISSING")

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
        encoder = preprocessing.LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    return data


def feature_engineering(data):
    # rolling demand features
    for shift in [28, 29, 30]:
        data[f"lag_t{shift}"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(shift)
        )

    for size in [7, 30]:
        data[f"rolling_std_t{size}"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(28).rolling(size).std()
        )

    for size in [7, 30, 90, 180]:
        data[f"rolling_mean_t{size}"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(28).rolling(size).mean()
        )

    data["rolling_skew_t30"] = data.groupby(["id"])["demand"].transform(
        lambda x: x.shift(28).rolling(30).skew()
    )
    data["rolling_kurt_t30"] = data.groupby(["id"])["demand"].transform(
        lambda x: x.shift(28).rolling(30).kurt()
    )

    # price features
    data["lag_price_t1"] = data.groupby(["id"])["sell_price"].transform(
        lambda x: x.shift(1)
    )
    data["price_change_t1"] = (data["lag_price_t1"] - data["sell_price"]) / (
        data["lag_price_t1"]
    )
    data["rolling_price_max_t365"] = data.groupby(["id"])["sell_price"].transform(
        lambda x: x.shift(1).rolling(365).max()
    )
    data["price_change_t365"] = (
        data["rolling_price_max_t365"] - data["sell_price"]
    ) / (data["rolling_price_max_t365"])

    data["rolling_price_std_t7"] = data.groupby(["id"])["sell_price"].transform(
        lambda x: x.rolling(7).std()
    )
    data["rolling_price_std_t30"] = data.groupby(["id"])["sell_price"].transform(
        lambda x: x.rolling(30).std()
    )
    data = data.drop(["rolling_price_max_t365", "lag_price_t1"], axis=1)

    # time features
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["week"] = data["date"].dt.week
    data["day"] = data["date"].dt.day
    data["dayofweek"] = data["date"].dt.dayofweek

    return data


def run_lgb(data):
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
        "lag_t28",
        "lag_t29",
        "lag_t30",
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
        "year",
        "month",
        "week",
        "day",
        "dayofweek",
    ]

    # going to evaluate with the last 28 days
    mask1 = data["date"] <= "2016-03-27"
    mask2 = data["date"] <= "2016-04-24"
    X_train = data[mask1]
    y_train = X_train.pop("demand")
    X_val = data[~mask1 & mask2]
    y_val = X_val.pop("demand")
    X_test = data[~mask2].drop("demand", axis=1)
    del data
    gc.collect()

    # define random hyperparammeters
    params = {
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

    train_set = lgb.Dataset(X_train[features], y_train)
    val_set = lgb.Dataset(X_val[features], y_val)

    del X_train, y_train

    model = lgb.train(params, train_set, valid_sets=[train_set, val_set], **fit_params)

    val_pred = model.predict(X_val[features])
    val_rmse = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
    print(f"RMSE:", val_rmse)
    y_pred = model.predict(X_test[features])
    return model, X_test.assign(demand=y_pred)


def make_submission(test, submission):
    preds = test[["id", "date", "demand"]]
    preds = pd.pivot(preds, index="id", columns="date", values="demand").reset_index()
    F_cols = ["F" + str(i + 1) for i in range(28)]
    preds.columns = ["id"] + F_cols

    assert preds[F_cols].isnull().sum().sum() == 0

    evals = submission[submission["id"].str.contains("evaluation")]
    vals = submission[["id"]].merge(preds, on="id")
    final = pd.concat([vals, evals])
    final.to_csv("submission.csv", index=False)


# %% [code]
data = transform(data)
data = feature_engineering(data)
data = reduce_mem_usage(data)
model, test = run_lgb(data)

# %% [code]
from mlflow_extend import plotting as mplt

imp_type = "gain"
features = model.feature_name()
importances = model.feature_importance(imp_type)
_ = mplt.feature_importance(features, importances, imp_type, limit=30)

# %% [code]
make_submission(test, submission)
