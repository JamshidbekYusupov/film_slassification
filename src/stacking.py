import pandas as pd 
import numpy as np 
import logging 
import os 
import joblib
from sklearn.compose import ColumnTransformer 
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import StackingClassifier



log_path = r'C:\animation_technique_classification\Logging\stacking.log'

logging.basicConfig(
    filename=log_path, 
    filemode='a', 
    level = logging.INFO, 
    format = '%(asctime)s-%(levelname)s-%(message)s'
)

logging.info('Stacking process has started')

class stacking_pipeline(BaseEstimator):

    def __init__(self, X_tain:pd.DataFrame,X_test:pd.DataFrame,
                  y_train:pd.DataFrame, y_test:pd.DataFrame,
                    base_learners, model_name, target:str, meta_model):
        # self.df = df.copy()
        # self.model_algorithm = model
        self.target = target
        self.model_name = model_name
        self.X_train = X_tain
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.preprocessor = None
        self.metrics = {}
        self.y_pred = None
        self.model = None
        self.meta_model = meta_model
        self.base_learners = base_learners

    try:

        def feature_preparation(self): 
 
            num_cols = self.X_train.select_dtypes(include = [np.number]).columns.to_list()
            cat_cols = self.X_train.select_dtypes(exclude = [np.number]).columns.to_list()

            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scalar', MinMaxScaler())
            ])

            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])

            self.preprocessor = ColumnTransformer([
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
            ])
            logging.info('Numeric and Categorical Pipelines are done ')
            return self
    except Exception as e:
        logging.error('Error while creating Column transforming')
    
    try:
        def fit(self):
            self.model = Pipeline([('processor', self.preprocessor),
                                   ('model', StackingClassifier(estimators=self.base_learners,
                                                                 final_estimator=self.meta_model))
])
            self.model.fit(self.X_train, self.y_train)

            logging.info(f'{self.model_name} is trained successfully')
    except Exception as e:
        logging.error(f'Error while training the model {e}')
        raise

    def model_saving(self): 
        try:
            out_dir  = r'C:\animation_technique_classification\Models\Stacking'
            os.makedirs(out_dir, exist_ok= True)
            out_path = os.path.join(out_dir, f'{self.model_name}.joblib')
            joblib.dump(self.model, out_path)
            logging.info(f'{self.model_name} was saved at {out_path}')
            return self
        except Exception as e:
            logging.error(f'Error while saving {self.model_name}: Error: {e}')
            raise
    def predicting(self):
        
            try:
                self.y_pred = self.model.predict(self.X_test)
                logging.info(f'Prediction is done for {self.model_name}')
                return self
            except Exception as e:
                logging.error(f'Error while predicting X_test for{self.model_name}')
                raise
    def evaluvating(self):
        try:
            self.metrics[f'{self.model_name}_accuracy'] = accuracy_score(self.y_test, self.y_pred)
            self.metrics[f'{self.model_name}_precision'] = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=True)
            self.metrics[f'{self.model_name}_recall'] =recall_score(self.y_test, self.y_pred, average='weighted', zero_division=True)
            self.metrics[f'{self.model_name}_f1'] = f1_score(self.y_test, self.y_pred, average='weighted', zero_division=True)
            return self
        
        except Exception as e: 
            logging.error(f'Error while calculating metrics for {self.model_name}')
            raise
    def metrics_saving(self):

        out_dir = r'C:\animation_technique_classification\Metrics\Stacking_Metrics'
        try: 

            df = pd.DataFrame([self.metrics])

            os.makedirs(out_dir, exist_ok= True)
            out_path = os.path.join(out_dir, f'{self.model_name}_metrics.csv')

            df.to_csv(out_path, index=False)

            logging.info(f'Metrics for {self.model_name} is saved at {out_dir}')

            return self
        except Exception as e: 
            logging.error(f'Error while saving metrics for {self.model_name}. Error: {e}')
            raise