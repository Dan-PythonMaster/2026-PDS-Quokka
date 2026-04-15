import os
import pandas as pd


def split_data(csv_path, train_pct, val_pct, seed, output_dir="./data/splits"):
    """
    Similar to the sci-kit-learn split function, but with csv outputs for easy inspection
    """
    if train_pct + val_pct >= 1.0:
        raise ValueError("percent values are too big")

    df = pd.read_csv(csv_path)
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(shuffled)
    n_train = int(n * train_pct)
    n_val = int(n * val_pct)

    train_df = shuffled.iloc[:n_train]
    val_df = shuffled.iloc[n_train:n_train + n_val]
    test_df = shuffled.iloc[n_train + n_val:]

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    return train_df, val_df, test_df
