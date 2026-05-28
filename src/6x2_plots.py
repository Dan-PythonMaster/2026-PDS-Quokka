import numpy as np
import matplotlib.pyplot as plt

from split_data import split_data
from model_suite import load_model


classes = ["BCC", "SCC", "MEL", "ACK", "NEV", "SEK"]
binary_lables = ["cancerous", "non-cancerous"]


output_dir = "./results/figures"

train_df, val_df, test_df = split_data(csv_path="./data/features.csv", train_pct=0.65, val_pct=0.20, seed=42)
forest = load_model("./results/models/model.pkl")

for split_name, df in [("val", val_df), ("test", test_df)]:
    drop_cols = [c for c in ["img_id", "diagnostic", "Unnamed: 0"] if c in df.columns]
    features = df.drop(columns=drop_cols)
    predictions = forest.predict(features)

    diag_matrix = np.zeros((6, 2), dtype=int)
    for true_diag, pred in zip(df["diagnostic"].values, predictions):
        if true_diag not in classes:
            continue
        diag_matrix[classes.index(true_diag), binary_lables.index(pred)] += 1
    plt.imshow(diag_matrix, cmap="Blues")
    plt.xticks(range(2), binary_lables, rotation=30, ha="right")
    plt.yticks(range(6), classes)
    plt.title(f"{split_name} diagnostic confusion")
    for i in range(6):
        for j in range(2):
            plt.text(j, i, diag_matrix[i, j], ha="center", va="center")
    plt.savefig(f"{output_dir}/{split_name}_confusion_diagnostic.png", bbox_inches="tight")
    plt.close()
