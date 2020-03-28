# %% [markdown]
# Python implementation of [M5 ForecasteR [0.57330]](https://www.kaggle.com/kailex/m5-forecaster-0-57330)


# %% [code]
import os
import gc
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# %% [code]
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)

# %% [code]
DAYS_PRED = 28


# %% [code]
def on_kaggle():
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


# %% [code]
def reduce_mem_usage(df, verbose=False):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    int_columns = df.select_dtypes(include=["int"]).columns
    float_columns = df.select_dtypes(include=["float"]).columns

    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

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
    prices = pd.read_csv(f"{INPUT_DIR}/sell_prices.csv").pipe(reduce_mem_usage)

    sales = pd.read_csv(f"{INPUT_DIR}/sales_train_validation.csv",).pipe(
        reduce_mem_usage
    )
    submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv").pipe(
        reduce_mem_usage
    )

    print("sales shape:", sales.shape)
    print("prices shape:", prices.shape)
    print("calendar shape:", calendar.shape)
    print("submission shape:", submission.shape)

    # calendar shape: (1969, 14)
    # sell_prices shape: (6841121, 4)
    # sales_train_val shape: (30490, 1919)
    # submission shape: (60980, 29)

    return sales, prices, calendar, submission


# %% [code]
sales, prices, calendar, submission = read_data()


# %% [code]
def reshape_sales(sales, submission):
    # melt sales data, get it ready for training
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # get product table.
    ids = sales[id_columns]

    melt_args = {
        "id_vars": id_columns,
        "var_name": "d",
        "value_name": "sales",
    }

    sales = sales.melt(**melt_args)
    sales = reduce_mem_usage(sales)

    # separate test dataframes.
    vals = submission[submission["id"].str.endswith("validation")]
    evals = submission[submission["id"].str.endswith("evaluation")]

    # change column names.
    vals.columns = ["id"] + [f"d_{d}" for d in range(1914, 1914 + DAYS_PRED)]
    evals.columns = ["id"] + [f"d_{d}" for d in range(1942, 1942 + DAYS_PRED)]

    # merge with product table
    evals["id"] = evals["id"].str.replace("_evaluation", "_validation")
    vals = vals.merge(ids, how="left", on="id")
    evals = evals.merge(ids, how="left", on="id")
    evals["id"] = evals["id"].str.replace("_validation", "_evaluation")

    vals = vals.melt(**melt_args)
    evals = evals.melt(**melt_args)

    sales["part"] = "train"
    vals["part"] = "validation"
    evals["part"] = "evaluation"

    sales = pd.concat([sales, vals, evals], axis=0)

    del vals, evals
    gc.collect()

    # Delete evaluation for now.
    sales = sales[sales["part"] != "evaluation"]

    return sales.drop("part", axis=1)


def merge_calendar(sales, calendar):
    usecols = ["date", "wm_yr_wk", "event_name_1", "snap_CA", "snap_TX", "snap_WI", "d"]
    return sales.merge(calendar[usecols], how="left", on="d")


def merge_prices(sales, prices):
    return sales.merge(prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])


def extract_num(ser):
    return ser.str.extract(r"(\d+)").astype(np.int16)


# %% [code]
sales = reshape_sales(sales, submission)
fillers = {
    "event_name_1": "",
    "event_type_1": "",
    "event_name_2": "",
    "event_type_2": "",
}

sales["d"] = extract_num(sales["d"])
calendar["d"] = extract_num(calendar["d"])

sales = merge_calendar(sales, calendar.fillna(fillers))
sales = merge_prices(sales, prices)

del calendar, prices
gc.collect()


# %% [code]
def add_sales_features(df):
    for lag in [7, 28, 29]:
        df[f"lag_{lag}"] = df.groupby(["id"])["sales"].transform(lambda x: x.shift(lag))

    for window in [7, 30, 90, 180]:
        df[f"roll_mean_28_{window}"] = df.groupby(["id"])["lag_28"].transform(
            lambda x: x.rolling(window).mean()
        )

    for window in [28]:
        df[f"roll_max_28_{window}"] = df.groupby(["id"])["lag_28"].transform(
            lambda x: x.rolling(window).max()
        )

        df[f"roll_var_28_{window}"] = df.groupby(["id"])["lag_28"].transform(
            lambda x: x.rolling(window).var()
        )

    return df


def add_price_features(df):
    df["price_shift_1"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.shift(1)
    )
    df["price_change_1"] = df["sell_price"] / df["price_shift_1"] - 1
    df["price_roll_max_365"] = df.groupby(["id"])["sell_price"].transform(
        lambda x: x.shift(1).rolling(365).max()
    )
    df["price_change_365"] = df["sell_price"] / df["price_roll_max_365"] - 1
    return df.drop(["price_shift_1", "price_roll_max_365"], axis=1)


def add_time_features(df):
    df["date"] = pd.to_datetime(df["date"])
    attrs = [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "dayofweek",
    ]

    for attr in attrs:
        df[attr] = getattr(df["date"].dt, attr)

    df["is_weekend"] = df["dayofweek"].isin([5, 6])

    return df


sales = sales.pipe(add_sales_features).pipe(add_price_features).pipe(add_time_features)


# %% [code]
def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)

    return df


# %% [code]
sales = encode_categorical(
    sales, ["item_id", "state_id", "dept_id", "cat_id", "event_name_1"]
)
sales = sales.drop(["store_id", "wm_yr_wk"], axis=1)
sales = reduce_mem_usage(sales)


# %% [code]
# 2011-01-29 ~ 2016-04-24 : d_1    ~ d_1913
# 2016-04-25 ~ 2016-05-22 : d_1914 ~ d_1941 (public)
# 2016-05-23 ~ 2016-06-19 : d_1942 ~ d_1969 (private)

is_train = sales["d"] < (1913 - DAYS_PRED)
is_val = (sales["d"] >= (1913 - DAYS_PRED)) & (sales["d"] < 1914)
is_test = sales["d"] >= 1914

drop_cols = ["id", "sales", "date", "d"]
X_train = sales[is_train].drop(drop_cols, axis=1)
y_train = sales[is_train]["sales"]

X_val = sales[is_val].drop(drop_cols, axis=1)
y_val = sales[is_val]["sales"]

X_test = sales[is_test].drop(drop_cols, axis=1)
id_date = sales[is_test][["id", "date"]]

del sales, is_train, is_val, is_test
gc.collect()


# %% [code]
def train_lgb(
    bst_params, fit_params, X_train, y_train, X_val, y_val, categorical_feature=None
):

    train_set = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical_feature
    )
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_feature)

    return lgb.train(
        bst_params,
        train_set,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        **fit_params,
    )


# %% [code]

bst_params = {
    "objective": "poisson",
    "metric": "rmse",
    "learning_rate": 0.075,
    "force_row_wise": True,
    "sub_feature": 0.8,
    "sub_row": 0.75,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "nthread": 4,
    "seed": 0,
}


fit_params = {
    "num_boost_round": 2000,
    "early_stopping_rounds": 400,
    "verbose_eval": 50,
}

categorical_feature = [
    "item_id",
    "state_id",
    "dept_id",
    "cat_id",
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "dayofweek",
    "is_weekend",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]

model = train_lgb(
    bst_params, fit_params, X_train, y_train, X_val, y_val, categorical_feature
)

del X_train, y_train, X_val, y_val
gc.collect()

# %% [code]
preds = model.predict(X_test)

del X_test
gc.collect()


# %% [code]
def make_submission(preds, submission):
    preds = preds.pivot(index="id", columns="date", values="sales").reset_index()
    preds.columns = ["id"] + ["F" + str(d + 1) for d in range(DAYS_PRED)]

    vals = submission[["id"]].merge(preds, how="inner", on="id")
    evals = submission[submission["id"].str.endswith("evaluation")]
    final = pd.concat([vals, evals])

    assert final.drop("id", axis=1).isnull().sum().sum() == 0
    assert final["id"].equals(submission["id"])

    final.to_csv("submission.csv", index=False)


# %% [code]
make_submission(id_date.assign(sales=preds), submission)
