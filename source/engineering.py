import pandas as pd
import numpy as np
import os, sys

sys.path.append(r"C:\animation_technique_classification")
raw_data = r'C:\animation_technique_classification\Data\Raw_Data\Raw_Data.csv'
df = pd.read_csv(raw_data)
target = 'Animation technique'

animation_techniques = ['Traditional', 'CG animation', 
                        'Flash animation', 'CGI animation',
                          'Computer']
from src.engineering import engineering

basic_eng = engineering(df, target, techniques=animation_techniques)

basic_eng.feature_prep().feature_adding().data_saving()