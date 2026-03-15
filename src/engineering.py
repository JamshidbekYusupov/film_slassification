import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.impute import SimpleImputer


log_path = r'C:\animation_technique_classification\Logging\engineering.log'

logging.basicConfig(
    filename=log_path,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s'
)
logging.info('Engineering has started')

class engineering:

    def __init__(self, df:pd.DataFrame, target: str, techniques: list):
        self.df = df.dropna(subset=['Release date'])
        self.target = target
        self.techniques = techniques
        self.X = self.df.drop(self.target, axis = 1)
        self.y = self.df[self.target].copy()
        self.mask = self.y.isin(self.techniques)
        self.y_filtered = self.y[self.mask]
        self.X_filtered = self.X[self.mask]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_filtered, self.y_filtered, 
                                                test_size=0.2, 
                                                random_state=42)
    
    def feature_prep(self):
        try:

            le = LabelEncoder()
            # X_train
            self.X_train['Release date'] = pd.to_datetime(self.X_train['Release date'], errors='coerce')
            self.X_train = self.X_train.dropna(subset=['Release date'])
            self.y_train = self.y_train[self.y_train.index.isin(self.X_train.index)]
            self.X_train = self.X_train.drop(columns=['Type', 'Notes', 'Title'])
            self.y_train = le.fit_transform(self.y_train)
            logging.info('Basic processing for Train set is DONE')

            # X_test
            self.X_test['Release date'] = pd.to_datetime(self.X_test['Release date'], errors='coerce')
            self.X_test = self.X_test.dropna(subset=['Release date'])
            self.y_test = self.y_test[self.y_test.index.isin(self.X_test.index)]
            self.X_test = self.X_test.drop(columns=['Type', 'Notes', 'Title'])
            self.y_test = le.transform(self.y_test)
            logging.info('Basic processing for Test set is DONE')
            return self
        
        except Exception as e:
            logging.error('Error while basic preprocessing')
            raise
    
    def feature_adding(self):
        try:
            # Adding features to x_train
            self.X_train['Release Year'] = self.X_train['Release date'].dt.year
            self.X_train['Release month'] = self.X_train['Release date'].dt.month
            self.X_train['Release day'] = self.X_train['Release date'].dt.day
            self.X_train['is_holiday'] = self.X_train['Release date'].dt.month.isin([11, 12])
            self.X_train['Duration_Minutes'] = self.X_train['Duration'].str.extract('(\d+)').astype(float)
            self.X_train['movie_category'] = pd.cut(self.X_train['Duration_Minutes'],
                                bins=[0, 45, 90, 150, 200],
                                labels=['short', 'standart', 'long', 'epic'])
            logging.info('Feature adding to X_train is DONE')
            
            # Adding features to x_test
            self.X_test['Release Year'] = self.X_test['Release date'].dt.year
            self.X_test['Release month'] = self.X_test['Release date'].dt.month
            self.X_test['Release day'] = self.X_test['Release date'].dt.day
            self.X_test['is_holiday'] = self.X_test['Release date'].dt.month.isin([11, 12])
            self.X_test['Duration_Minutes'] = self.X_test['Duration'].str.extract('(\d+)').astype(float)
            self.X_test['movie_category'] = pd.cut(self.X_test['Duration_Minutes'],
                                bins=[0, 45, 90, 150, 200],
                                labels=['short', 'standart', 'long', 'epic'])
            logging.info('Feature adding to X_test is DONE')


            # dropping unncessary columns after using them
            self.X_train = self.X_train.drop(columns=['Year', 'Release date', 'Duration'])
            logging.info('Feature adding is done successfully')

            return self
        except Exception as e:
            logging.error(f'Error while feature adding {e}')
            raise

    def oversampling(self):
        try:
            encoder = LabelEncoder()
            for col in self.X_train.columns:
             if self.X_train[col].isnull().sum() >= 1 and self.X_train[col].dtype == "str":
                 self.X_train[col] = self.X_train[col].fillna(self.X_train[col].mode()[0])
             if self.X_train[col].isnull().sum() >= 1 and self.X_train[col].dtype != "str":
                self.X_train[col] = self.X_train[col].fillna(value=self.X_train[col].mean())

            for col in self.X_train.columns:
             if self.X_train[col].dtype == "str":
                  self.X_train[col] = encoder.fit_transform(self.X_train[col])

            # cat_cols = self.X_train.select_dtypes(exclude=[np.number]).columns.to_list()

            # for col in cat_cols:
            #     self.X_train[col]  = encoder.fit_transform(self.X_train[col])

            smote = SMOTE(random_state=42, k_neighbors=1)

            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

            logging.info(f'Over sampling with SMOTE is done. Categories: {Counter(self.y)}')

            return self
        except Exception as e:
            logging.error(f'Error while Over Sampling. Error: {e}')
            raise

    def data_saving(self):
        try:
            # X_train saving 
            data_path = r'C:\animation_technique_classification\Data\basic_eng'
            os.makedirs(data_path, exist_ok=True)
            path = os.path.join(data_path, 'X_train.csv')
            self.X_train.to_csv(path, index=False)

            logging.info(f'X_train is saved at {data_path}')

            # X_test saving 
            data_path = r'C:\animation_technique_classification\Data\basic_eng'
            os.makedirs(data_path, exist_ok=True)
            path = os.path.join(data_path, 'X_test.csv')
            self.X_test.to_csv(path, index=False)

            logging.info(f'X_test is saved at {data_path}')

             # y_train
            path = os.path.join(data_path, 'y_train.csv')
            self.y_train = pd.DataFrame(self.y_train, columns=[self.target])
            self.y_train.to_csv(path, index=False)
            logging.info(f'y_train is saved at {data_path}')
            # y_test
            path = os.path.join(data_path, 'y_test.csv')
            self.y_test = pd.DataFrame(self.y_test, columns=[self.target])
            self.y_test.to_csv(path, index=False)
            logging.info(f'y_test is saved at {data_path}')
            
            return self
        except Exception as e:
            logging.error(f'Error while saving data at {data_path}')
            raise