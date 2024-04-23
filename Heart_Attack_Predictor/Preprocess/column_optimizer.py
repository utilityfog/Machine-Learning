import os
import pandas as pd
import numpy as np

from statsmodels.discrete.discrete_model import Probit
from statsmodels.api import OLS
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

from typing import List

from .raw_dataframe_preprocessor import FINAL_RANDHIE_REGRESSORS, FINAL_RANDHIE_Y

class ColumnRearranger:
    def return_optimal_rearrangement(self, df_left, df_right, X_left=FINAL_RANDHIE_REGRESSORS, X_right=[]) -> pd.DataFrame:
        """
        function that receives both the left and right dataframes as well as a list of regressors for each, rearranges the columns of the right dataframe
        such that it best aligns with the column arrangement of the left dataframe and then returns the rearranged right dataframe.
        """
        # This function is necessary because vectorization of rows depends on column arrangement because each column represents a dimension
        n, m = len(X_left), len(X_right)  # Number of columns specified in each dataset
        if n > m:
            raise ValueError("There are more regressors specified in the left DataFrame than in the right DataFrame.")
        
        # Initialize cost matrix
        cost_matrix = np.zeros((n, m))
        
        # Calculate negative correlation coefficients to fill the cost matrix
        for i, col_left in enumerate(X_left):
            for j, col_right in enumerate(X_right):
                if df_right[col_right].dtype.kind in 'oi' and df_left[col_left].dtype.kind in 'oi':
                    # Handle non-numeric columns or use appropriate encoding/transformations
                    continue
                model = LinearRegression().fit(df_right[[col_right]], df_left[[col_left]])
                prediction = model.predict(df_right[[col_right]])
                corr, _ = pearsonr(prediction.flatten(), df_left[col_left])
                cost_matrix[i, j] = -corr  # Use negative correlation because we minimize in Hungarian
        
        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create a new DataFrame with columns rearranged according to the optimal assignment
        rearranged_columns = [X_right[idx] for idx in col_ind]
        rearranged_df_right = df_right[rearranged_columns].copy()

        return rearranged_df_right