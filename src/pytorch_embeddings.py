# %% [markdown]
# # This kernel is based on:
# ## - [M5 Forecast: Keras with Categorical Embeddings V2](https://www.kaggle.com/mayer79/m5-forecast-keras-with-categorical-embeddings-v2)
# ## - [m5-baseline](https://www.kaggle.com/harupy/m5-baseline)


# %% [code]
import os
import gc
import warnings

import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)
register_matplotlib_converters()
sns.set()


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
def reduce_mem_usage(df, verbose=False):
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
                    df[col] = df[col].astype(np.float32)
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

NUM_ITEMS = sales.shape[0]  # 30490
DAYS_PRED = submission.shape[1] - 1  # 28

# %% [markdown]
# As [@kaushal2896](https://www.kaggle.com/kaushal2896) suggested in [this comment](https://www.kaggle.com/harupy/m5-baseline#770558), encode the categorical columns before merging to prevent the notebook from crashing even with the full dataset.


# %% [code]
def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("__MISSING__"))
    return df


calendar = encode_categorical(
    calendar, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
).pipe(reduce_mem_usage)

sales = encode_categorical(
    sales, ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
).pipe(reduce_mem_usage)

prices = encode_categorical(prices, ["item_id", "store_id"]).pipe(reduce_mem_usage)


# %% [code]
def extract_num(ser):
    return ser.str.extract(r"(\d+)").astype(np.int16)


def reshape_sales(
    sales, submission, d_thresh=0, verbose=True,
):
    # melt sales data, get it ready for training
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # get product table.
    product = sales[id_columns]

    sales = sales.melt(id_vars=id_columns, var_name="d", value_name="demand",)
    sales = reduce_mem_usage(sales)

    if verbose:
        print("melted")
        display(sales)

    # separate test dataframes.
    vals = submission[submission["id"].str.endswith("validation")]
    evals = submission[submission["id"].str.endswith("evaluation")]

    # change column names.
    vals.columns = ["id"] + [f"d_{d}" for d in range(1914, 1914 + DAYS_PRED)]
    evals.columns = ["id"] + [f"d_{d}" for d in range(1942, 1942 + DAYS_PRED)]

    # merge with product table
    evals["id"] = evals["id"].str.replace("_evaluation", "_validation")
    vals = vals.merge(product, how="left", on="id")
    evals = evals.merge(product, how="left", on="id")
    evals["id"] = evals["id"].str.replace("_validation", "_evaluation")

    if verbose:
        print("validation")
        display(vals)

        print("evaluation")
        display(evals)

    vals = vals.melt(id_vars=id_columns, var_name="d", value_name="demand")
    evals = evals.melt(id_vars=id_columns, var_name="d", value_name="demand")

    sales["part"] = "train"
    vals["part"] = "validation"
    evals["part"] = "evaluation"

    data = pd.concat([sales, vals, evals], axis=0)

    del sales, vals, evals

    data["d"] = extract_num(data["d"])
    data = data[data["d"] >= d_thresh]

    # delete evaluation for now.
    data = data[data["part"] != "evaluation"]

    gc.collect()

    if verbose:
        print("data")
        display(data)

    return data


def merge_calendar(data, calendar):
    calendar = calendar.drop(["weekday", "wday", "month", "year"], axis=1)
    return data.merge(calendar, how="left", on="d")


def merge_prices(data, prices):
    return data.merge(prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])


# %% [code]
data = reshape_sales(sales, submission, d_thresh=1941 - 365 * 3)
del sales
gc.collect()

calendar["d"] = extract_num(calendar["d"])
data = merge_calendar(data, calendar)
del calendar
gc.collect()

data = merge_prices(data, prices)
del prices
gc.collect()

data = reduce_mem_usage(data)


# %% [code]
def add_demand_features(df):
    for diff in [0, 1, 2]:
        shift = DAYS_PRED + diff
        df[f"shift_t{shift}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(shift)
        )

    for window in [7, 30, 60, 90, 180]:
        df[f"rolling_std_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).std()
        )

    for window in [7, 30, 60, 90, 180]:
        df[f"rolling_mean_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(window).mean()
        )

    df["rolling_skew_t30"] = df.groupby(["id"])["demand"].transform(
        lambda x: x.shift(DAYS_PRED).rolling(30).skew()
    )
    df["rolling_kurt_t30"] = df.groupby(["id"])["demand"].transform(
        lambda x: x.shift(DAYS_PRED).rolling(30).kurt()
    )
    return df


def add_price_features(df):
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


def add_time_features(df, dt_col):
    df[dt_col] = pd.to_datetime(df[dt_col])
    attrs = [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "dayofweek",
        "is_year_end",
        "is_year_start",
        "is_quarter_end",
        "is_quarter_start",
        "is_month_end",
        "is_month_start",
    ]

    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        df[attr] = getattr(df[dt_col].dt, attr).astype(dtype)

    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
    return df


# %% [code]
data = add_demand_features(data).pipe(reduce_mem_usage)
data = add_price_features(data).pipe(reduce_mem_usage)
dt_col = "date"
data = add_time_features(data, dt_col).pipe(reduce_mem_usage)
data = data.sort_values("date")

print("start date:", data[dt_col].min())
print("end date:", data[dt_col].max())
print("data shape:", data.shape)


# %% [code]
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

num_cols = [
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "sell_price",
    # demand features.
    "shift_t28",
    "shift_t29",
    "shift_t30",
    "rolling_std_t7",
    "rolling_std_t30",
    "rolling_std_t60",
    "rolling_std_t90",
    "rolling_std_t180",
    "rolling_mean_t7",
    "rolling_mean_t30",
    "rolling_mean_t60",
    "rolling_mean_t90",
    "rolling_mean_t180",
    "rolling_skew_t30",
    "rolling_kurt_t30",
    # price features
    "price_change_t1",
    "price_change_t365",
    "rolling_price_std_t7",
    "rolling_price_std_t30",
    # time features.
    "year",
    "month",
    "week",
    "day",
    "dayofweek",
    "is_weekend",
]


for col in num_cols:
    data[col] = data[col].fillna(data[col].median())


# %% [code]
# prepare training and test data.
# 2011-01-29 ~ 2016-04-24 : d_1    ~ d_1913
# 2016-04-25 ~ 2016-05-22 : d_1914 ~ d_1941 (public)
# 2016-05-23 ~ 2016-06-19 : d_1942 ~ d_1969 (private)

features = cat_cols + num_cols
is_train = data["d"] < 1914

# Attach "d" to X_train for cross validation.
X_train = data[is_train][["d"] + features].reset_index(drop=True)
y_train = data[is_train]["demand"].astype(np.float32).reset_index(drop=True)
X_test = data[~is_train][features].reset_index(drop=True)

# Keep "id" and "date" to use later.
id_date = data[~is_train][["id", "date"]].reset_index(drop=True)

del data
gc.collect()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# %% [code]
class Model(nn.Module):
    def __init__(self, embedding_sizes, dense_size):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(*size) for size in embedding_sizes]
        )

        emb_size_sum = sum([size[1] for size in embedding_sizes])
        self.linears = nn.Sequential(
            nn.Linear(emb_size_sum + dense_size, 150),
            nn.ReLU(),
            nn.Linear(150, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x_cat, x_num):
        embeddings = []
        for idx, emb in enumerate(self.embeddings):
            embeddings.append(emb(x_cat[:, idx].long()))
        x = torch.cat([x_num.float()] + embeddings, dim=1)

        return self.linears(x)


# %% [code]
embedding_sizes = {
    "item_id": (3049, 3),
    "dept_id": (7, 1),
    "store_id": (10, 1),
    "cat_id": (3, 1),
    "state_id": (3, 1),
    "event_name_1": (31, 1),
    "event_type_1": (5, 1),
    "event_name_2": (5, 1),
    "event_type_2": (5, 1),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cat_cols = list(embedding_sizes.keys())
model = Model(embedding_sizes.values(), len(num_cols))
model.to(device)


# %% [code]
class TabularDataset(Dataset):
    def __init__(self, X, cat_cols, y=None):
        self.n = X.shape[0]
        self.X = X
        self.y = y

        self.cat_cols = cat_cols
        self.num_cols = [c for c in X.columns if c not in cat_cols]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.y is not None:
            return [
                self.X[self.cat_cols].iloc[idx].values,
                self.X[self.num_cols].iloc[idx].values,
                self.y.iloc[idx],
            ]
        else:
            return [
                self.X[self.cat_cols].iloc[idx].values,
                self.X[self.num_cols].iloc[idx].values,
            ]


# %% [code]
train_set = TabularDataset(X_train[cat_cols + num_cols], cat_cols, y_train)
test_set = TabularDataset(X_test[cat_cols + num_cols], cat_cols)

batch_size = 64
train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=4)


# %% [code]
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return ((y_true - y_pred) ** 2).mean().sqrt()


# %% [code]
num_epochs = 5
criterion = RMSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1)

loss_epoch = []

for epoch in range(num_epochs):
    running_loss = 0
    for X_cat, X_num, y in train_loader:
        X_cat = X_cat.to(device)
        X_num = X_num.to(device)
        y = y.to(device)

        # Forward Pass.
        preds = model(X_cat, X_num)
        loss = criterion(preds, y)

        # Backward Pass and Optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print("RMSE:", loss.item())

    running_loss /= len(train_loader)
    loss_epoch.append(running_loss)

print(loss_epoch)

# %% [code]
model.eval()

preds = []
for X_cat, X_num in test_loader:
    with torch.no_grad():
        X_cat = X_cat.to(device)
        X_num = X_num.to(device)

        pred = model(X_cat, X_num).detach()
        preds.append(pred.squeeze().cpu().numpy())

preds = np.hstack(preds)


# %% [code]
def make_submission(test, submission):
    preds = test[["id", "date", "demand"]]
    preds = preds.pivot(index="id", columns="date", values="demand").reset_index()
    preds.columns = ["id"] + ["F" + str(d + 1) for d in range(DAYS_PRED)]

    vals = submission[["id"]].merge(preds, how="inner", on="id")
    evals = submission[submission["id"].str.endswith("evaluation")]
    final = pd.concat([vals, evals])

    assert final.drop("id", axis=1).isnull().sum().sum() == 0
    assert final["id"].equals(submission["id"])

    final.to_csv("submission.csv", index=False)


# %% [code]
make_submission(id_date.assign(demand=preds), submission)
