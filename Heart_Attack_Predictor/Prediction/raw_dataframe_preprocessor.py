import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
from statsmodels.api import OLS
import statsmodels.api as sm

class RANDHIE:
    def preprocess(self, df_path):
        # Load the dataset
        df = pd.read_csv(df_path)

        # Constructing necessary variables for the model
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
    
class heart:
    def preprocess(self):
        df = None
        return df