from src.split_data import split_data
import src.model_suite
import pandas as pd

def main(features_path, prediction_results_path, model_path, load_model, baseline):
    """
    Docstring for main
    
    :param features_path: Path to the features csv used as input to the model (e.g. ./data/features.csv).
    :param prediction_results_path: Path to save the output predictions of the model (e.g. ./result/predictions/predictions_MODEL.csv).
    :param model_path: Path to save or load the trained model (e.g. ./result/predictions/predictions_MODEL.csv).
    :param load_model: Boolean to train the model and save it to model_path if False, load it from model_path if True. 
    """
    
    # load dataset CSV file

    # split the dataset into training and testing sets.
    train_df, val_df, test_df = split_data(csv_path=features_path, train_pct=0.65, val_pct=0.20, seed=42)
    
    if baseline: 
        train_df, val_df, test_df = [i.drop(columns=["contrast_diff", "contrast_ratio", "contrast_standardized"]) for i in [train_df, val_df, test_df]]


    if load_model:
        # load the model
        model = src.model_suite.load_model(model_path)

    else:
        # train the classifier (using logistic regression as an example)
        # save the model.
        src.model_suite.train_classifier(train_df, val_df, model_path=model_path)
        # load the model
        model = src.model_suite.load_model(model_path)

    # test the classifier and write test results to CSV.
    src.model_suite.test_model(model, test_df, prediction_results_path)

    



if __name__ == "__main__":
    features_path = "./data/features.csv"
    prediction_results_path = "./results/predictions/predictions_MODEL.csv"
    model_path = "./results/models/model.pkl"
    load_model = False
    baseline = False

    main(features_path, prediction_results_path,model_path,load_model,baseline)