from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

RAW_DIR = '/data/jihye_data/BenchmarkDataset/Gas_Sensor_Array_Drift_Dataset'
RANDOM_STATE = 0

class GasSensorArrayDriftDataset():
    def __init__(self, dataset_id=1):
        df = self.load_data(RAW_DIR, dataset_id)
        X, y = self.get_X_y(df)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        
    def load_data(self, data_dir, dataset_id):
        filepath = os.path.join(data_dir, 'batch{}.dat'.format(dataset_id))
        print('Loading {}..'.format(filepath))
        df = pd.read_table(filepath, sep="\s+", header=None)
        for col in df.columns[1:]:
            df[col] = df[col].apply(lambda x: x.split(':')[-1])
        return df
    
    def get_X_y(self, df):
        X = df[df.columns[1:]].values.astype(np.float32)
        X = np.expand_dims(X, axis=0)
        y = df[0].values.astype(np.float32)
        return X, y