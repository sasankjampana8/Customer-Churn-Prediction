import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # Explicitly enable IterativeImputer
from sklearn.impute import IterativeImputer
import pickle as pkl



class DataPrep:
    def __init__(self, data, artifacts_path):
        self.data = data
        self.artifacts_path = artifacts_path
        X = self.data.drop(['Churn'], axis=1)
        Y = self.data['Churn']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.3, random_state=42
        )
        
    def save_artifacts(self, feature, encoder):
        path = f'{self.artifacts_path}/{feature}_estimator.pkl'
        with open(path, 'wb') as f: 
            pkl.dump(encoder, f)
    
    def label_encode(self):
        """Label encode specified categorical columns."""
        cat_cols = ["county", "city", "marital_status"]
        self.X_train[cat_cols] = self.X_train[cat_cols].fillna("missing")
        self.X_test[cat_cols] = self.X_test[cat_cols].fillna("missing")
        
        for col in cat_cols:
            label_encoder = LabelEncoder()
            label_encoder.fit(self.X_train[col])
            self.X_train[col] = label_encoder.transform(self.X_train[col])
            self.X_test[col] = self.X_test[col].apply(
                lambda s: "Others" if s not in label_encoder.classes_ else s
            )
            self.X_test[col] = label_encoder.transform(self.X_test[col])
            self.save_artifacts(f'{col}_label_encoder', label_encoder)

    def home_market_value(self):
        """Calculate average home market value from a range."""
        def calculate_average(value):
            try:
                start, end = map(float, value.split(' - '))
                return (start + end) / 2
            except Exception:
                return np.nan

        self.X_train['average_home_market_value'] = self.X_train['home_market_value'].apply(calculate_average)
        self.X_test['average_home_market_value'] = self.X_test['home_market_value'].apply(calculate_average)

    def impute(self):
        """Impute missing values using Iterative Imputer."""
        imputer = IterativeImputer(random_state=42)
        self.X_train = pd.DataFrame(imputer.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_test = pd.DataFrame(imputer.transform(self.X_test), columns=self.X_test.columns)
        self.save_artifacts("mice_imputer", imputer)
        
    def account_status(self):
        """Create binary feature for account suspension status."""
        self.X_train['account_status'] = self.X_train['acct_suspd_date'].notnull().astype(int)
        self.X_test['account_status'] = self.X_test['acct_suspd_date'].notnull().astype(int)
    
    def drop_columns(self):
        """Drop unnecessary columns."""
        cols = ['individual_id', 'address_id', 'cust_orig_date', 'date_of_birth', 'acct_suspd_date', 'home_market_value', 'state']
        self.X_train.drop(cols, axis=1, inplace=True)
        self.X_test.drop(cols, axis=1, inplace=True)
        
    def addressid(self):
        """Create count of occurrences for each address_id."""
        self.X_train['address_id_count'] = self.X_train.groupby('address_id')['address_id'].transform('count')
        self.X_test['address_id_count'] = self.X_test.groupby('address_id')['address_id'].transform('count')

    def preprocess(self):
        """Run all preprocessing steps."""
        self.label_encode()
        self.account_status()
        self.home_market_value()
        self.addressid()
        self.drop_columns()
        self.impute()
        return self.X_train, self.X_test, self.Y_train, self.Y_test
