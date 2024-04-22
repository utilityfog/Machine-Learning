import pandas as pd
import numpy as np

from statsmodels.discrete.discrete_model import Probit
from statsmodels.api import OLS
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from .globals import RANDHIE_CATEGORICAL_VARIABLES, RANDHIE_NUMERIC_VARIABLES

def standardize_dataframe(df, numerical_columns, exclude_columns):
    """
    A function to standardize specified numerical values in a DataFrame using z-score normalization,
    excluding specified columns.
    
    Parameters:
        df (pandas DataFrame): The input DataFrame.
        numerical_columns (list): A list of numerical column names to standardize.
        exclude_columns (list): A list of column names to exclude from standardization.
        
    Returns:
        pandas DataFrame: A DataFrame with standardized numerical values.
    """
    # Exclude columns specified in exclude_columns
    columns_to_standardize = [col for col in numerical_columns if col not in exclude_columns]
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler to the selected columns and transform the values
    standardized_values = scaler.fit_transform(df[columns_to_standardize])
    
    # Create a new DataFrame with the standardized values and the same index and columns as the original DataFrame
    standardized_df = pd.DataFrame(standardized_values, index=df.index, columns=columns_to_standardize)
    
    # Combine the standardized numerical columns with non-numerical columns from the original DataFrame
    for col in df.columns:
        if col not in columns_to_standardize:
            standardized_df[col] = df[col]
    
    return standardized_df

def encode_categorical(df, categorical_vars):
    """
    A function to perform one-hot encoding for categorical variables in a DataFrame.
    
    Parameters:
        df (pandas DataFrame): The input DataFrame.
        categorical_columns (list): A list of column names containing categorical variables to be one-hot encoded.
        
    Returns:
        pandas DataFrame: A DataFrame with one-hot encoded categorical variables.
    """
    # Extract categorical variables
    categorical_df = df[categorical_vars]
    
    # Perform one-hot encoding for categorical variables, drop first ensures there is no multicolinearity
    result = pd.get_dummies(categorical_df, drop_first=True)
    
    return result

def replace_encoded_categorical(df, encoded_categorical_df, categorical_columns):
    """
    A function to replace original categorical columns in a DataFrame with one-hot encoded columns.
    
    Parameters:
        df (pandas DataFrame): The original DataFrame.
        encoded_categorical_df (pandas DataFrame): The DataFrame with one-hot encoded categorical variables.
        categorical_columns (list): A list of column names containing original categorical variables.
        
    Returns:
        pandas DataFrame: A DataFrame with original categorical columns replaced by one-hot encoded columns.
    """
    # Drop original categorical columns from the original DataFrame
    df = df.drop(columns=categorical_columns)
    
    # Concatenate the original DataFrame with the one-hot encoded DataFrame
    df = pd.concat([df, encoded_categorical_df], axis=1)
    return df

def remove_nan_or_inf(df: pd.DataFrame):
    """
    Removes rows from a DataFrame that contain NaN or infinite values in any column.

    Parameters:
    - df: pd.DataFrame

    Returns:
    - pd.DataFrame: DataFrame with rows containing NaN or infinite values removed.
    """
    # Remove rows with NaN values
    df_cleaned = df.dropna()

    # Remove rows with infinite values
    df_cleaned = df_cleaned[~(df_cleaned.isin([np.inf, -np.inf]).any(axis=1))]

    return df_cleaned

class RANDHIE:
    def original_preprocess(self, df_path):
        """
        Method that pre-processes the randhie.csv dataset exactly as it was preprocessed in the demand for medical care paper
        
        Returns: pd.DataFrame
        """
        df = pd.read_csv(df_path)
        print(f"Raw DataFrame: {df.head()}")

        # Re-Constructing Variables for the Four Equations from the Paper: ""
        df['positive_med_exp'] = (df['meddol'] > 0).astype(int)  # 1 if positive medical expenses, else 0
        df['positive_inpatient_exp'] = ((df['inpdol'] > 0) & (df['meddol'] > 0)).astype(int)  # 1 if positive inpatient expenses and positive medical use, else 0
        df['only_outpatient_exp'] = ((df['inpdol'] == 0) & (df['meddol'] > 0)).astype(int)  # 1 if only outpatient expenses and positive medical use, else 0
        df['log_med_exp'] = np.where(df['meddol'] > 0, np.log(df['meddol']), 0)  # Log transformation for positive expenses
        df['log_inpatient_exp'] = np.where(df['inpdol'] > 0, np.log(df['inpdol']), 0)  # Log transformation for positive inpatient expenses

        # Define independent variables based on the paper's model and available data
        X_vars = ['xage', 'linc', 'coins', 'black', 'female', 'educdec']
        X = df[X_vars]
        X = sm.add_constant(X)  # Adds a constant term to the predictor

        # Equation 1: Probit model for zero versus positive medical expenses
        model_1 = Probit(df['positive_med_exp'], X).fit()

        # Equation 2: Probit model for having zero versus positive inpatient expense, given positive use of medical services
        df_pos_med_exp = df[df['positive_med_exp'] == 1]  # Filter for positive medical use
        model_2 = Probit(df_pos_med_exp['positive_inpatient_exp'], X.loc[df_pos_med_exp.index]).fit()

        # Equation 3: OLS regression for log of positive medical expenses if only outpatient services are used
        df_only_outpatient_exp = df[df['only_outpatient_exp'] == 1]
        model_3 = OLS(df_only_outpatient_exp['log_med_exp'], X.loc[df_only_outpatient_exp.index]).fit()

        # Equation 4: OLS regression for log of medical expenses for those with any inpatient expenses
        df_pos_inpatient_exp = df[df['positive_inpatient_exp'] == 1]
        model_4 = OLS(df_pos_inpatient_exp['log_inpatient_exp'], X.loc[df_pos_inpatient_exp.index]).fit()

        # Print summaries of the models
        print("Model 1: Probit model for zero versus positive medical expenses")
        print(model_1.summary())
        print("\nModel 2: Probit model for having zero versus positive inpatient expense, given positive use of medical services")
        print(model_2.summary())
        print("\nModel 3: OLS regression for log of positive medical expenses if only outpatient services are used")
        print(model_3.summary())
        print("\nModel 4: OLS regression for log of medical expenses for those with any inpatient expenses")
        print(model_4.summary())
        
    def improved_preprocess(self, df_path):
        """
        Method that pre-processes the randhie.csv dataset such that we can test at the individual patient level which predictors are statistically significant
        
        Returns: pd.DataFrame
        """
        # Read
        df = pd.read_csv(df_path)
        print(f"Raw DataFrame: {df.head()}")
        
        # Remove rows with NaN or inf values
        df_cleaned = remove_nan_or_inf(df)
        print(f"DataFrame after removing NaN and inf values: {df_cleaned.head()}")
        
        # Average the numeric column values at the patient (zper) level since we are not interested in time and the RANDHIE experiment has 5 individual year observations for each patient
        collapsed_df = self.average_by_unique_patient(df_cleaned, "zper") # Only average numeric columns when collapsing!!
        print(f"AVERAGED: {collapsed_df.head()}")
        
        # Re-Constructing Variables for the Four Equations from the Paper (must be done before standardization as it is affected by sign): ""
        collapsed_df['is_positive_med_exp'] = (collapsed_df['meddol'] > 0).astype(int)  # 1 if positive medical expenses, else 0
        collapsed_df['is_positive_inpatient_exp'] = ((collapsed_df['inpdol'] > 0) & (collapsed_df['meddol'] > 0)).astype(int)  # 1 if positive inpatient expenses and positive medical use, else 0
        collapsed_df['is_only_outpatient_exp'] = ((collapsed_df['inpdol'] == 0) & (collapsed_df['meddol'] > 0)).astype(int)  # 1 if only outpatient expenses and positive medical use, else 0
        collapsed_df['log_med_exp'] = np.where(collapsed_df['meddol'] > 0, np.log(collapsed_df['meddol']), 0)  # Log transformation for positive expenses
        collapsed_df['log_inpatient_exp'] = np.where(collapsed_df['inpdol'] > 0, np.log(collapsed_df['inpdol']), 0)  # Log transformation for positive inpatient expenses
        paper_variables_categorical = ['is_positive_med_exp', 'is_positive_inpatient_exp', 'is_only_outpatient_exp']
        paper_variables_numeric = ['log_med_exp', 'log_inpatient_exp']
        print(f"processed_df's generated columns: {collapsed_df[paper_variables_numeric + paper_variables_categorical].head()}")
        
        # Standardize Numeric Columns: standardized_df = standardize_df(avg_df)
        new_numeric_columns = RANDHIE_NUMERIC_VARIABLES + paper_variables_numeric
        new_categorical_columns = RANDHIE_CATEGORICAL_VARIABLES + paper_variables_categorical
        standardized_df = standardize_dataframe(collapsed_df, new_numeric_columns, new_categorical_columns)
        print(f"STANDARDIZED - NEW: {standardized_df.head()}")
        # Add plan without standardization
        standardized_df['plan'] = collapsed_df['plan']
        
        # Combine standardized_df's child and fchild into one categorical column as they are redundant
        
        # One Hot Encoding categorical variables
        encoded_df = encode_categorical(standardized_df, new_categorical_columns)
        print(f"ONE HOT ENCODED (CATEGORICAL VARS): {encoded_df.head()}")
        
        # Replacing the randhie df's categorical variables with one hot encoded variables
        processed_df = replace_encoded_categorical(standardized_df, encoded_df, new_categorical_columns)
        print(f"PROCESSED: {processed_df.head()}")

        # Define independent variables based on the paper's model and available data: 
            # Excluding variables that are known to be endogenous (e.g. if someone makes a lot of hospital visits, obviously it will have a positive causal relationship with their quantity demanded for medical care even if there are confounding variables that may affect the number of times they visit the hospital)
            # Excluding variables that are a deterministic function of another to prevent perfect multicolinearity; no inclusion of both linc and income
            # preprocess child into one categorical variable column
        X_list = ['plan', 'tookphys', 'xage', 'educdec', 'time', 'disea', 'physlm', 'mdeoff', 'lfam', 'lpi', 'logc', 'xghindx', 'linc', 'lnum', 'site', 'black', 'female', 'mhi', 'hlthg', 'hlthf', 'hlthp']
        X = processed_df[X_list]
        X = sm.add_constant(X)  # Adds an intercept term
        
        # Four equation model according to paper

        # Equation 1: Probit model for zero versus positive medical expenses
        model_1 = Probit(processed_df['is_positive_med_exp'], X).fit()

        # Equation 2: Probit model for having zero versus positive inpatient expense, given positive use of medical services
        df_pos_med_exp = processed_df[processed_df['is_positive_med_exp'] == 1]  # Filter for positive medical use
        model_2 = Probit(df_pos_med_exp['is_positive_inpatient_exp'], X.loc[df_pos_med_exp.index]).fit()

        # Equation 3: OLS regression for log of positive medical expenses if only outpatient services are used
        df_only_outpatient_exp = processed_df[processed_df['is_only_outpatient_exp'] == 1]
        model_3 = OLS(df_only_outpatient_exp['log_med_exp'], X.loc[df_only_outpatient_exp.index]).fit()

        # Equation 4: OLS regression for log of medical expenses for those with any inpatient expenses
        df_pos_inpatient_exp = processed_df[processed_df['is_positive_inpatient_exp'] == 1]
        model_4 = OLS(df_pos_inpatient_exp['log_inpatient_exp'], X.loc[df_pos_inpatient_exp.index]).fit()

        # Print summaries of the models
        print("Model 1: Probit model for zero versus positive medical expenses")
        print(model_1.summary())
        print("\nModel 2: Probit model for having zero versus positive inpatient expense, given positive use of medical services")
        print(model_2.summary())
        print("\nModel 3: OLS regression for log of positive medical expenses if only outpatient services are used")
        print(model_3.summary())
        print("\nModel 4: OLS regression for log of medical expenses for those with any inpatient expenses")
        print(model_4.summary())
        
        return processed_df
        
    def average_by_unique_patient(self, df, id_column):
        """
        A function to average attributes of a time series dataset by a unique ID.
        
        Parameters:
            df (pandas DataFrame): The input dataset.
            id_column (str): The name of the column containing unique IDs.
            attribute_columns (list): A list of column names containing attributes to be averaged.
            
        Returns:
            pandas DataFrame: A DataFrame with averaged rows, indexed by unique ID.
        """
        attributes = df.columns.tolist()
        result = df.groupby(id_column)[attributes].mean()
        return result
    
class HEART:
    def preprocess(self):
        df = None
        return df