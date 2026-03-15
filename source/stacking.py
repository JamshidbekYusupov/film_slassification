import pandas as pd
import numpy as np 
import os, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

sys.path.append(r'C:\animation_technique_classification')

from src.stacking import stacking_pipeline

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

base_learners = [
    ('dt', DecisionTreeClassifier(max_depth=5)),
    ('svc', SVC()),
    ('rf',LogisticRegression())
]

meta_model = RandomForestClassifier()
name = "Voting"


basic_pipe = stacking_pipeline(X_tain=X_train,X_test=X_test, y_train=y_train,
                                y_test=y_test, base_learners = base_learners,
                                  model_name=name, target=target, meta_model = meta_model)

basic_pipe.feature_preparation()
basic_pipe.fit()
basic_pipe.model_saving()
basic_pipe.predicting()
basic_pipe.evaluvating()
basic_pipe.metrics_saving()