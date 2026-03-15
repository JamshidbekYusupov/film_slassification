import pandas as pd
import numpy as np 
import os, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


sys.path.append(r'C:\animation_technique_classification')

from src.tuning import tuning_pipeline

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

model = RandomForestClassifier(random_state=42)
name = "RF"

param_grid = {
    'model__n_estimators': [50, 100, 150, 200, 250, 300],
    'model__max_depth': [None, 2, 5, 10, 15, 20],
    'model__min_samples_leaf': [1, 2, 3, 4, 5], 
    'model__min_samples_split': [2, 3, 4, 5, 6, 8],
    'model__bootstrap': [True, False], 
    'model__class_weight': ['balanced', 'balanced_subsample', None] 
}
tuning_pipe = tuning_pipeline(X_train=X_train,X_test=X_test, y_train=y_train,
                                y_test=y_test, model=model, model_name=name, target=target)

tuning_pipe.feature_preparation()
# tuning_pipe.fit()
tuning_pipe.tune_grid_search(param_grid)
tuning_pipe.model_saving()
tuning_pipe.predicting()
tuning_pipe.evaluvating()
tuning_pipe.metrics_saving()