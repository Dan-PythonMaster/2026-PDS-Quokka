import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
try:
    from src.split_data import split_data
except:
    from split_data import split_data

cancerous = {"BCC", "SCC", "MEL"}


class LoadedModel:
    def __init__(self, forest):
        self.forest = forest
        self.classes_ = forest.classes_

    def predict(self, features):
        return self.forest.predict(features)


def train_classifier(train_df, val_df, model_path="./results/models/model.pkl"):
    '''
    Classifier where new features are easy to add, has a primitive gridsearch and saves the model 
    '''
    drop_cols = [c for c in ["img_id", "diagnostic", "Unnamed: 0"] if c in train_df.columns]
    train_features = train_df.drop(columns=drop_cols).select_dtypes(include="number")
    train_labels = np.where(train_df["diagnostic"].isin(cancerous), "cancerous", "non-cancerous")

    drop_cols = [c for c in ["img_id", "diagnostic", "Unnamed: 0"] if c in val_df.columns]
    val_features = val_df.drop(columns=drop_cols).select_dtypes(include="number")
    val_labels = np.where(val_df["diagnostic"].isin(cancerous), "cancerous", "non-cancerous")

    best_forest = None
    best_val_accuracy = 0

    for trees in (100, 200, 300, 400):
        for depth in (None, 8, 16, 24, 32):
            forest = RandomForestClassifier(
                n_estimators=trees,
                max_depth=depth,
                class_weight="balanced",
                random_state=6,
            )
            forest.fit(train_features, train_labels)
            val_accuracy = forest.score(val_features, val_labels)

            if val_accuracy > best_val_accuracy:
                best_forest = forest
                best_val_accuracy = val_accuracy

    print(f"Best val accuracy: ",best_val_accuracy)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(val_labels, best_forest.predict(val_features), cmap="Blues")
    plt.savefig(model_path.replace(".pkl", "_val_cm.png"), bbox_inches="tight")
    plt.close()

    with open(model_path, "wb") as f:
        pickle.dump(best_forest, f)
    return best_forest


def load_model(model_path):
    with open(model_path, "rb") as model:
        return LoadedModel(pickle.load(model))


def test_model(forest, test_df, prediction_results_path="./results/predictions/val_predictions_MODEL.csv"):
    '''
    Summarizes the model, makes CSV and generates a plot for deeper understanding of the features
    '''
    drop_cols = [c for c in ["img_id", "diagnostic", "Unnamed: 0"] if c in test_df.columns]
    test_features = test_df.drop(columns=drop_cols).select_dtypes(include="number")
    test_labels = np.where(test_df["diagnostic"].isin(cancerous), "cancerous", "non-cancerous")

    predictions = forest.predict(test_features)
    print(f"Accuracy: {accuracy_score(test_labels, predictions)}")
    print(classification_report(test_labels, predictions))
   ##################################open question skin tone split
    if "skin_tone_proxy" in test_df.columns:
        test_df = test_df.copy()
        test_df["tertile"] = pd.qcut(
            test_df["skin_tone_proxy"],
            q=3,
            labels=["dark", "medium", "light"]
        )
        print("\nSkin tone tertile distribution:")
        print(test_df["tertile"].value_counts())
    ####################

    underlying = forest.forest
    cancer_index = list(underlying.classes_).index("cancerous")
    non_cancer_index = list(underlying.classes_).index("non-cancerous")
    probabilities = underlying.predict_proba(test_features)

    results_df = pd.DataFrame({
        "label": test_labels,
        "prediction": predictions,
        "p_cancerous": probabilities[:, cancer_index],
        "p_non_cancerous": probabilities[:, non_cancer_index],
        "correct": predictions == test_labels,
    })
    
    os.makedirs(os.path.dirname(prediction_results_path), exist_ok=True)
    results_df.to_csv(prediction_results_path, index=False)
    
    ConfusionMatrixDisplay.from_predictions(test_labels, predictions, cmap="Blues")
    plt.savefig(prediction_results_path.replace(".csv", "_test_cm.png"), bbox_inches="tight")
    plt.close()

    ####################

    shap_values = shap.TreeExplainer(underlying).shap_values(test_features)
    shap_cancer = shap_values[:, :, cancer_index]

    mean_shap = pd.Series(shap_cancer.mean(axis=0), index=test_features.columns).sort_values(key=abs)

    fig, (ax_cm, ax_bars) = plt.subplots(1, 2, figsize=(13, 6))

    ConfusionMatrixDisplay.from_predictions(test_labels, predictions, cmap="Blues", ax=ax_cm, colorbar=False)
    mean_shap.plot.barh(ax=ax_bars, color=["red" if v > 0 else "blue" for v in mean_shap])
  
    ax_bars.set_title("toward cancerous vs toward non-cancerous")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_df, val_df, test_df = split_data(csv_path="./data/features.csv", train_pct=0.6, val_pct=0.2, seed=6)
    train_classifier(train_df, val_df)
    loaded_model = load_model("./results/models/model.pkl")
    test_model(loaded_model, val_df)
