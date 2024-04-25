import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import torch

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

script_directory = os.path.dirname(os.path.abspath(__file__))
projects_directory = os.path.dirname(script_directory)
sys.path.append(projects_directory)

from Preprocess import raw_dataframe_preprocessor, column_optimizer
from VQ_VAE import Encoder, VQVAE
from HNSW import Row_Matcher

# HYPERPARAMETERS
BATCH_SIZE = 512
NUM_TRAINING_UPDATES = 2000

NUM_HIDDENS = 128
NUM_RESIDUAL_HIDDENS = 32
NUM_RESIDUAL_LAYERS = 2

EMBEDDING_DIM = 39
NUM_EMBEDDINGS = 512

COMMITMENT_COST = 0.25

DECAY = 0.99

LEARNING_RATE = 1e-3

def main():
    # randhie dataset path
    randhie_path = os.getcwd()+"/PVM/Datasets/randhie.csv"
    
    # Initialize RANDHIE class instance
    randhie = raw_dataframe_preprocessor.RANDHIE()
    
    # Pre-processed randhie dataset
    randhie_preprocessed, randhie_X = randhie.improved_preprocess(randhie_path)
    
    # heart dataset path
    heart_path = os.getcwd()+"/PVM/Datasets/heart_attack_prediction_dataset.csv"
    
    # Initialize HEART class instance
    heart = raw_dataframe_preprocessor.HEART()
    
    heart_preprocessed, heart_X = heart.preprocess(heart_path)
    
    column_rearranger = column_optimizer.ColumnRearranger()
    
    # Reduce row number of heart table to match that of randhie via bootstrapping
    heart_X = column_rearranger.bootstrap_to_match(randhie_X, heart_X)
    
    # pre-rearrangement
    average_correlation_pre = column_rearranger.compute_average_correlation(randhie_X, heart_X)
    print(f"pre operation average correlation: {average_correlation_pre}")
    
    # Rearrange columns of the right table such that the average correlation between every column i from the left table and every column j from the right table where i=j is maximized
    heart_X_rearranged = column_rearranger.return_optimal_rearrangement(randhie_X, heart_X)
    
    # post-rearrangement
    average_correlation_post = column_rearranger.compute_average_correlation(randhie_X, heart_X_rearranged)
    print(f"post operation average correlation: {average_correlation_post}")
    
    column_rearranger.visualize_comparison(average_correlation_pre, average_correlation_post)
    
    # Update global df
    raw_dataframe_preprocessor.update_rearranged_final_predictor_dataframe(heart_X_rearranged)
    
    # Visualize if rearrangement was done correctly
    raw_dataframe_preprocessor.save_dataframe(heart_X_rearranged, os.getcwd()+"/PVM/Datasets", "heart_preprocessed_X_rearranged.csv")
    
    encoder = Encoder.DataFrameEncoder()
    # Train the models
    encoder.train_and_assign_models()
    # Save the models
    encoder.save_model(encoder.randhie_model, 'randhie_model.pth')
    encoder.save_model(encoder.heart_model, 'heart_model.pth')
    # Encode randhie and heart dataframes
    encoded_randhie_df, encoded_heart_df = encoder.load_and_encode_dataframes(randhie_X, heart_X_rearranged)
    
    raw_dataframe_preprocessor.save_dataframe(encoded_randhie_df, os.getcwd()+"/PVM/Datasets", "randhie_predictors.csv")
    print(f"{encoded_randhie_df.head()}")
    
    raw_dataframe_preprocessor.save_dataframe(encoded_heart_df, os.getcwd()+"/PVM/Datasets", "heart_predictors.csv")
    print(f"{encoded_heart_df.head()}")
    
    ############################
    randhie_predictors_path = os.getcwd()+"/PVM/Datasets/randhie_predictors.csv"
    randhie_predictors = encoded_randhie_df
    # pd.read_csv(randhie_predictors_path)
    
    heart_predictors_path = os.getcwd()+"/PVM/Datasets/heart_predictors.csv"
    heart_predictors = encoded_heart_df
    # pd.read_csv(heart_predictors_path)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    randhie_model = VQVAE.Model(39, NUM_HIDDENS, EMBEDDING_DIM).to(device)
    randhie_model.load_state_dict(torch.load('randhie_model.pth'))
    randhie_model.eval()
    
    heart_model = VQVAE.Model(54, NUM_HIDDENS, EMBEDDING_DIM).to(device)
    heart_model.load_state_dict(torch.load('heart_model.pth'))
    heart_model.eval()
    
    ### Row Matching Logic
    
    row_matcher = Row_Matcher.RowMatcher()

    def match_rows(randhie_df, heart_df):
        row_matcher = Row_Matcher.RowMatcher()
        return row_matcher.retrieve_similar(randhie_df, heart_df)
    
    # Perform row matching and store results
    combined_data = match_rows(encoded_randhie_df, encoded_heart_df)

    # Save and display results
    combined_data.to_csv(os.getcwd() + "/PVM/Datasets/merged_predictors.csv")
    print(combined_data.head())
    
    # FINAL REGRESSION!
    # Final regression
    final_randhie_regressors, final_heart_regressors, final_randhie_y, final_heart_y = raw_dataframe_preprocessor.return_final_variables()
    
    randhie_y = randhie_preprocessed[final_randhie_y]
    
    heart_y = heart_preprocessed[final_heart_y]
    
    predictors = final_randhie_regressors + ['Stress Level', 'Sedentary Hours Per Day', 'Obesity_1', 'Cholesterol']
    print(f"Using predictors: {predictors}")  # Debug print

    regression_results = {}

    for target in randhie_y.columns:
        print(f"Running regression for target: {target}")  # Debug print

        X = combined_data[predictors]
        y = randhie_y[target]
        X_const = sm.add_constant(X)  # Add constant for intercept

        X_train, X_test, y_train, y_test = train_test_split(X_const, y, test_size=0.2, random_state=42)

        model = sm.OLS(y_train, X_train)
        results = model.fit()

        y_pred = results.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        regression_results[target] = {
            'Summary': results.summary(),
            'MSE': mse
        }

    for target, data in regression_results.items():
        print(f"Results for {target}:")
        print(data['Summary'])
        print(f"Mean Squared Error: {data['MSE']}")


if __name__ == "__main__":
    main()