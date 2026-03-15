import pandas as pd
import numpy as np
import os
import logging

logging_path = r'C:\animation_technique_classification\Logging\data_loader.log'

logging.basicConfig(
    filename= logging_path,
    filemode='a',
    level= logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s'
)

logging.info('Data Loading has started')

class data_loader:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(r'C:\animation_technique_classification\Scraping\animated_films_2010.csv')
    
    def getting_data(self):
        try: 
            years = [2011, 2012, 2013, 2014, 2015,
                    2016, 2017, 2018, 2019,2020, 
                    2021, 2022, 2023, 2024, 2025]
            
            for year in years:

                url = rf'C:\animation_technique_classification\Scraping\animated_films_{year}.csv'

                sdf = pd.read_csv(url)

                self.df = pd.concat([self.df, sdf])

                logging.info(f'Data for the year {year} is concated')
            return self
        except Exception as e:
            logging.error(f'Error while concating the dataset of {year}. Error: {e}')
        
    def saving_data(self):

        try:
            os.makedirs(self.path, exist_ok=True)

            path = os.path.join(self.path, 'Raw_Data.csv')

            self.df.to_csv(path, index = False)

            logging.info(f'Data is saved at {self.path}')

            return self
        
        except Exception as e:
            logging.error(f"Error while saving the data. Error: {e}")




