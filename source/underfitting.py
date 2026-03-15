import pandas as pd
import numpy as np 
import os, sys
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier

sys.path.append(r'C:\animation_technique_classification')

from src.overfitting import overfitting_pipeline

data_path = r'C:\air_quality\Data\Raw_data\raw_data.csv'
X_train = pd.read_csv(r'C:\animation_technique_classification\Data\basic_eng\X_train.csv')
X_test = pd.read_csv(r'C:\animation_technique_classification\Data\basic_eng\X_test.csv')
y_train = pd.read_csv(r'C:\animation_technique_classification\Data\basic_eng\y_train.csv')
y_test = pd.read_csv(r'C:\animation_technique_classification\Data\basic_eng\y_test.csv')

str_cols = X_train.select_dtypes(include=['str', 'string', 'object']).columns

# Convert them to object
X_train[str_cols] = X_train[str_cols].astype(object)
X_test[str_cols] = X_test[str_cols].astype(object)

target = 'Animation technique'

models = {
    "RF_Under": RandomForestClassifier(max_depth=1, random_state=42), 
    "DT_Under": DecisionTreeClassifier(max_depth=1, random_state=42)
}

for name, model in models.items(): 

    
    over_pipe = overfitting_pipeline(X_tain=X_train,X_test=X_test, y_train=y_train,
                                y_test=y_test, model=model, model_name=name, target=target)

    over_pipe.feature_preparation()
    over_pipe.fit()
    over_pipe.model_saving()
    over_pipe.predicting()
    over_pipe.evaluvating()
    over_pipe.metrics_saving()