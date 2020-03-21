import os

import pandas as pd


INPUT_DIR = "input/m5-forecasting-accuracy"
OUTPUT_DIR = "input/sample"


def read_and_sample():
    sales = (
        pd.read_csv(f"{INPUT_DIR}/sales_train_validation.csv")
        .groupby("cat_id")
        .head(5)
        .reset_index(drop=True)
    )

    item_ids = sales["item_id"].tolist()
    ids = sales["id"].tolist()

    cal = pd.read_csv(f"{INPUT_DIR}/calendar.csv")

    prices = pd.read_csv(f"{INPUT_DIR}/sell_prices.csv")
    prices = prices[prices["item_id"].isin(item_ids)]

    sbm = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")
    sbm = sbm[
        sbm["id"].isin(ids)
        | sbm["id"].str.replace("_evaluation", "_validation").isin(ids)
    ]
    return sales, prices, cal, sbm


def main():
    sales, prices, cal, sbm = read_and_sample()

    print(f"sell_prices shape: {prices.shape}")
    print(f"sales_train_val shape: {sales.shape}")
    print(f"calendar shape: {cal.shape}")
    print(f"submission shape: {sbm.shape}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sales.to_csv(os.path.join(OUTPUT_DIR, "sales_train_validation.csv"), index=False)
    cal.to_csv(os.path.join(OUTPUT_DIR, "calendar.csv"), index=False)
    prices.to_csv(os.path.join(OUTPUT_DIR, "sell_prices.csv"), index=False)
    sbm.to_csv(os.path.join(OUTPUT_DIR, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    main()
