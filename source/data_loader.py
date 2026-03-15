import pandas as pd 
import numpy as np
import os, sys

sys.path.append(r'C:\animation_technique_classification')

data_path = r'C:\animation_technique_classification\Data\Raw_Data'

from src.data_loader import data_loader


dl = data_loader(data_path)
dl.getting_data().saving_data()





