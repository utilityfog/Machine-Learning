import os
import sys
import pandas as pd

script_directory = os.path.dirname(os.path.abspath(__file__))
projects_directory = os.path.dirname(script_directory)
sys.path.append(projects_directory)

from Preprocess import raw_dataframe_preprocessor, column_optimizer

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
    
    # Visualize if rearrangement was done correctly
    raw_dataframe_preprocessor.save_dataframe(heart_X_rearranged, os.getcwd()+"/PVM/Datasets", "heart_preprocessed_X_rearranged.csv")

if __name__ == "__main__":
    main()