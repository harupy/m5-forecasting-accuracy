# %% [markdown]
# # Credits
#
# * [First R notebook](https://www.kaggle.com/kailex/m5-forecaster-v2)
# * [Python translation](https://www.kaggle.com/kneroma/m5-forecast-v2-python)
# * [M5 First Public Notebook Under 0.50](https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50)

# %% [code]
from functools import reduce
from datetime import datetime, timedelta
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb


# %% [code]
h = 28
max_lags = 57
tr_last = 1913
fday = datetime(2016, 4, 25)


# %% [code]
# INPUT_DIR = "input/sample"
INPUT_DIR = "../input/m5-forecasting-accuracy"


def read_prices():
    dtypes = {
        "store_id": "category",
        "item_id": "category",
        "wm_yr_wk": "int16",
        "sell_price": "float32",
    }
    prices = pd.read_csv(f"{INPUT_DIR}/sell_prices.csv", dtype=dtypes)

    for col, dtype in dtypes.items():
        if dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")

    return prices


def read_calendar():
    dtypes = {
        "event_name_1": "category",
        "event_name_2": "category",
        "event_type_1": "category",
        "event_type_2": "category",
        "weekday": "category",
        "wm_yr_wk": "int16",
        "wday": "int16",
        "month": "int16",
        "year": "int16",
        "snap_CA": "float32",
        "snap_TX": "float32",
        "snap_WI": "float32",
    }
    cal = pd.read_csv(f"{INPUT_DIR}/calendar.csv", dtype=dtypes)
    cal["date"] = pd.to_datetime(cal["date"])

    for col, dtype in dtypes.items():
        if dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")

    return cal


def create_sales(is_train=True, nrows=None, first_day=1200):
    start_day = max(1 if is_train else tr_last - max_lags, first_day)
    d_cols = [f"d_{day}" for day in range(start_day, tr_last + 1)]
    id_cols = ["id", "item_id", "dept_id", "store_id", "cat_id", "state_id"]
    dtype = {c: "float32" for c in d_cols}
    dtype.update({c: "category" for c in id_cols if c != "id"})

    sales = pd.read_csv(
        f"{INPUT_DIR}/sales_train_validation.csv",
        nrows=nrows,
        usecols=id_cols + d_cols,
        dtype=dtype,
    )

    for col in id_cols:
        if col != "id":
            sales[col] = sales[col].cat.codes.astype("int16")

    if not is_train:
        for day in range(tr_last + 1, tr_last + h + 1):
            sales[f"d_{day}"] = np.nan

    prices = read_prices()
    cal = read_calendar()

    return (
        sales.melt(id_vars=id_cols, var_name="d", value_name="sales")
        .merge(cal, on="d", copy=False)
        .merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)
    )


# %% [code]
def add_lag_features(df):
    lags = [7, 28]
    for lag in lags:
        col = f"lag_{lag}"
        df[col] = df[["id", "sales"]].groupby("id")["sales"].shift(lag)

    windows = [7, 28]
    for win in windows:
        for lag in lags:
            col = f"rmean_{lag}_{win}"
            lag_col = f"lag_{lag}"
            df[col] = (
                df[["id", lag_col]]
                .groupby("id")[lag_col]
                .transform(lambda x: x.rolling(win).mean())
            )
    return df


def add_time_features(df):
    time_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in time_features.items():
        if date_feat_name in df.columns:
            df[date_feat_name] = df[date_feat_name].astype("int16")
        else:
            df[date_feat_name] = getattr(df["date"].dt, date_feat_func).astype("int16")

    return df


def apply_funcs(df, funcs):
    reduce(lambda x, f: f(x), funcs, df)


# %% [code]
sales = create_sales(is_train=True, first_day=350)

# %% [code]
sales.head()

# %% [code]
funcs = [add_lag_features, add_time_features]
sales = apply_funcs(sales, funcs)

# %% [code]
sales.head()

# %% [code]
sales = sales.dropna()

# %% [code]
cat_features = [
    "item_id",
    "dept_id",
    "store_id",
    "cat_id",
    "state_id",
    "event_name_1",
    "event_name_2",
    "event_type_1",
    "event_type_2",
]
drop_cols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]
X_train = sales.drop(drop_cols, axis=1)
y_train = sales["sales"]


# %% [code]
train_set = lgb.Dataset(
    X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False,
)


# %% [code]
del sales, X_train, y_train
gc.collect()

# %% [code]
bst_params = {
    "objective": "poisson",
    "metric": "rmse",
    "force_row_wise": True,
    "learning_rate": 0.075,
    "sub_row": 0.75,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "verbosity": 1,
    "num_leaves": 128,
    "min_data_in_leaf": 100,
}

train_params = {
    "num_boost_round": 1200,
    "verbose_eval": 50,
}


# %% [code]
model = lgb.train(bst_params, train_set, **train_params)


# %% [code]
alphas = [1.028, 1.023, 1.018]
weights = [1 / len(alphas)] * len(alphas)

for idx, (alpha, weight) in enumerate(zip(alphas, weights)):

    te = create_sales(is_train=False)
    F_cols = [f"F{i}" for i in range(1, 29)]

    for tdelta in range(0, 28):
        day = fday + timedelta(days=tdelta)
        mask = (te["date"] >= day - timedelta(days=max_lags)) & (te["date"] <= day)
        tst = te[mask].copy()
        tst = apply_funcs(tst, funcs)
        tst = tst.loc[tst["date"] == day, model.feature_name()]
        te.loc[te["date"] == day, "sales"] = alpha * model.predict(tst)

    te_sub = te.loc[te["date"] >= fday, ["id", "sales"]].copy()
    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount() + 1]
    te_sub = te_sub.set_index(["id", "F"]).unstack()["sales"][F_cols].reset_index()
    te_sub = te_sub.fillna(0.0).sort_values("id").reset_index(drop=True)

    if idx == 0:
        sub = te_sub
        sub[F_cols] *= weight
    else:
        sub[F_cols] += te_sub[F_cols] * weight

    print(idx, alpha, weight)


sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace(r"validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv("submission.csv", index=False)

# %% [code]
sub.head(10)


# %% [code]
sub.shape
