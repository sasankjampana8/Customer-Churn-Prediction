import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os 


class DataPrep:
    def __init__(self, data):
        self.data = data
        Y = self.data['']
        X = X.drop([''], axis=1)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    
    def label_encode(self,):
        encoder = LabelEncoder()
        
    def one_hot_encode(self,):
        encoder = OneHotEncoder()
        
    def home_market_value(self,):
        pass
    
    def date_col(self,):
        self.X_train
        self.X_val
    
    
    def preprocess(self,):
        return self.X_train, self.X_val, self.Y_train, self.Y_val
        