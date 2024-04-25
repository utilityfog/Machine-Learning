import os
import sys
import pandas as pd
import torch

script_directory = os.path.dirname(os.path.abspath(__file__))
projects_directory = os.path.dirname(script_directory)
sys.path.append(projects_directory)

from Preprocess import raw_dataframe_preprocessor, column_optimizer
from VQ_VAE import Encoder, VQVAE
from HNSW import Row_Matcher

# HYPERPARAMETERS
BATCH_SIZE = 1024
NUM_TRAINING_UPDATES = 15000

NUM_HIDDENS = 128
NUM_RESIDUAL_HIDDENS = 32
NUM_RESIDUAL_LAYERS = 2

EMBEDDING_DIM = 64
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
    
    print(f"{encoded_randhie_df.head()}")
    
    print(f"{encoded_heart_df.head()}")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    row_matcher = Row_Matcher.RowMatcher()
    
    randhie_model = VQVAE.Model(39, NUM_HIDDENS,
                NUM_EMBEDDINGS, EMBEDDING_DIM, 
                COMMITMENT_COST).to(device)
    randhie_model.load_state_dict(torch.load('randhie_model.pth'))
    randhie_model.eval()
    
    heart_model = VQVAE.Model(54, NUM_HIDDENS,
                NUM_EMBEDDINGS, EMBEDDING_DIM, 
                COMMITMENT_COST).to(device)
    heart_model.load_state_dict(torch.load('heart_model.pth'))
    heart_model.eval()
    
    # Create an HNSW index for the heart DataFrame using the heart model
    heart_index = row_matcher.create_index(heart_model, encoded_heart_df, device=device)

    # Retrieve similar rows
    merged_df = row_matcher.retrieve_similar(encoded_randhie_df, encoded_heart_df, randhie_model, heart_index, device=device)

    # Display results
    print(merged_df.head())
    
    # Save
    raw_dataframe_preprocessor.save_dataframe(merged_df, os.getcwd()+"/PVM/Datasets", "merged_predictors.csv")

if __name__ == "__main__":
    main()