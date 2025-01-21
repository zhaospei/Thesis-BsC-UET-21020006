# require a lot cpu & gpu ram: 6x ram cpu, 24gb ram gpu
import pandas as pd
import torch


from utils import training_classify


def filter_df(df):
    df = df[df["memory"] > 0]
    df = df[df["time"] > 0]
    return df

layers = [1, 4, 8, 12, 16, 20, 24, 28, 32]
fields = [
    "first_token_embedding",
    "last_token_embedding",
    "first_token_code_embedding",
    "last_token_code_embedding",
]


label_type = "nonf"
print(label_type)
labels = ["memory",'time']

for lb in labels:
    print("debug")
    print("*" * 50)
    for layer in layers:
        for field in fields:
            print("###", layer, field)
            print("\t", "-" * 33)
            test_df = pd.read_parquet(f"data/codellama/train_nonf/{layer}/test.parquet")
            train_df = pd.read_parquet(
                f"data/codellama/train_nonf/{layer}/train.parquet"
            )
            train_df = filter_df(train_df)
            test_df = filter_df(test_df)

            training_classify(
                train_df, test_df, field, lb, f"codellama_{lb}_{layer}_{label_type}"
            )
