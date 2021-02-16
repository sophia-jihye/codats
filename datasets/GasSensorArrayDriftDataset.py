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
            
        def list_to_2d_array(list_x, sensor_num):
            x = np.array(list_x)
            x_new = x.reshape((sensor_num, -1))
            return x_new   # (?, 16, 8)
        
        faults = df[df.columns[0]]
        df = df[df.columns[1:]]
        results = df.apply(lambda x: x.dropna().tolist(), axis=1).tolist()
        matrix_df = pd.DataFrame({'sensor*time': results, 'fault': faults})
        matrix_df['fault'] = matrix_df['fault'].apply(lambda x: x-1)
        matrix_df['sensors'] = matrix_df['sensor*time'].apply(lambda x: list_to_2d_array(x, 16))
        return matrix_df
    
    def get_X_y(self, df):
        X = np.expand_dims(df['sensors'].values[0], axis=0)
        for arr in tqdm(df['sensors'].values[1:]):
            X = np.vstack([X, np.expand_dims(arr, axis=0)]).astype(np.float32)   # (161, 8, 16)
        y = df['fault'].values.astype(np.float32)
        return X, y