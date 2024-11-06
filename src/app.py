import pandas as pd 
import numpy as np 
from datapreperation import DataPrep
import os

def main():
    data = pd.read_csv("/home/jampanasasank/Desktop/Customer Churn Prediction/Customer-Churn-Prediction/AllData/archive (6)/autoinsurance_churn.csv")
    artifacts_path = '/home/jampanasasank/Desktop/Customer Churn Prediction/Customer-Churn-Prediction/model'
    prep = DataPrep(data=data)
    X_train, X_val, Y_train, Y_val = prep.preprocess()
    