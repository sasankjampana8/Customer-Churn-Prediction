import os
import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
import pickle as pkl 



class Preprocess:
    def __init__(self, path):
        self.model_path = path
        self.estimators = dict()
        try:
            for file in os.listdir(self.model_path):
                if "pkl" in file:
                    self.estimators[file.split(".")[0]] = pkl.load(open(os.path.join(self.model_path, file), 'rb'))
                elif "model" in file:
                    continue
            print("Loaded the estimators...")
            print(self.estimators)
        except Exception as e:
            print(f"Error loading the estimators: {e}")
            
    def label_encode(self, data):
        try:
            cat_cols = ["county", "city", "marital_status"]
            for col in cat_cols:
                col_name = f"{col}_label_encoder_estimator"
                encoder = self.estimators[col_name]
                data[col].fillna("missing", inplace=True)
                data[col] = data[col].map(
                    lambda s: "Others" if s not in encoder.classes_ else s
                ) 
                data[col] = encoder.transform(data[col])
            
            return data
                
        
        except Exception as e:
            print(f"Error label encoding categorical columns: {e}")
            
    
    def drop_columns(self, data):
        data.drop(['individual_id', 'address_id', 'cust_orig_date', 'date_of_birth', 'acct_suspd_date', 'home_market_value', 'state'], axis=1, inplace=True)
        return data
    
    def addressid(self, data):
        """Create count of occurrences for each address_id."""
        
        data['address_id_count'] = data.groupby('address_id')['address_id'].transform('count')
        return data
    
    def home_market_value(self, data):
        """Calculate average home market value from a range."""
        def calculate_average(value):
            try:
                start, end = map(float, value.split(' - '))
                return (start + end) / 2
            except Exception:
                return np.nan

        data['average_home_market_value'] = data['home_market_value'].apply(calculate_average)
        
        return data
    
    def impute(self, data):
        """Impute missing values in the data."""
        imputer_name = "mice_imputer_estimator"
        cols = data.columns
        imputer = self.estimators[imputer_name]
        data = imputer.transform(data)
        
        return pd.DataFrame(data, columns=cols)
    
    def preprocess(self, data):
        data = self.label_encode(data)
        data = self.home_market_value(data)
        data = self.addressid(data)
        data = self.drop_columns(data)
        data = self.impute(data)
        return data
        
        
        
        