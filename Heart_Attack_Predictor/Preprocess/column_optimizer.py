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
    def return_optimal_rearrangement(self, df_left, df_right, X_left, X_right) -> pd.DataFrame:
        """
        function that receives both the left and right dataframes as well as a list of regressors for each, rearranges the columns of the right dataframe
        such that it best aligns with the column arrangement of the left dataframe and then returns the rearranged right dataframe.
        """
        # This function is necessary because vectorization of rows depend on column arrangement because each column represents a dimension
        n, m = df_left.shape[1], df_right.shape[1]  # Number of columns in each dataset
        cost_matrix = np.zeros((n, m))

        # Calculate correlation coefficients and fill the cost matrix
        for i in range(n):
            for j in range(m):
                model = LinearRegression().fit(df_right.iloc[:, [j]], df_left.iloc[:, i])
                prediction = model.predict(df_right.iloc[:, [j]])
                corr, _ = pearsonr(prediction.flatten(), df_left.iloc[:, i])
                cost_matrix[i, j] = -corr  # Negative because we need to maximize

        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Resulting optimal assignment from heart to randhie
        optimal_assignment = list(zip(df_left.columns[row_ind], df_right.columns[col_ind]))
        print(optimal_assignment)