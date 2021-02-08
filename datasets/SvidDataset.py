from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

SVID_RAW_DIR = '/data/jihye_data/Samsung_time-series/SVID/rawdata'
RANDOM_STATE = 0

def get_numpy_from_nonfixed_2d_array(aa, fixed_length, padding_value=0):
    rows = []
    for a in aa:
        rows.append(np.pad(a, (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length])
    return np.concatenate(rows, axis=0).reshape(-1, fixed_length)

class SvidDataset():
    def __init__(self, recipe_num=1):
        df = self.load_svid(SVID_RAW_DIR, recipe_num)
        X, y = self.get_X_y(df)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        
    def load_svid(self, data_dir, recipe_num):
        filepath = os.path.join(data_dir, 'RECIPE{}_STEP{:02}.csv')
        fault_filepath = os.path.join(data_dir, 'RECIPE{}_fault.csv')
        
        def list_to_2d_array(list_x):
            x = np.array(list_x)
            x_new = x.reshape((65, -1))
            return x_new

        print('Recipe', recipe_num)
        faults = pd.read_csv(fault_filepath.format(recipe_num), names=['fault']).fault.values
        svid_df = pd.DataFrame({'fault': faults})
        for i in tqdm(range(1, 26)):
            df = pd.read_csv(filepath.format(recipe_num, i), header=None).transpose()
            results = df.apply(lambda x: x.dropna().tolist(), axis=1).tolist()
            sensor_by_time_colname = 'STEP{:02}_sensor*time'.format(i)
            temp_df = pd.DataFrame({sensor_by_time_colname: results})
            temp_df['STEP{:02}_sensors'.format(i)] = temp_df[sensor_by_time_colname].apply(lambda x: list_to_2d_array(x))
            temp_df.drop(columns=[sensor_by_time_colname], inplace=True)
            svid_df = pd.concat([svid_df, temp_df], axis=1)
        svid_df['Recipe_num'] = recipe_num
        
        svid_df['STEP01-25_sensors'] = svid_df['STEP01_sensors']
        svid_df.drop(columns=['STEP01_sensors'], inplace=True)
        for i in range(2, 26):
            colname = 'STEP{:02}_sensors'.format(i)
            svid_df['STEP01-25_sensors'] = svid_df.apply(lambda x: np.hstack([x['STEP01-25_sensors'], x[colname]]),axis=1)
            svid_df.drop(columns=[colname], inplace=True)
            
        fixed_length= 1952   # Source domain과 Unlabeled target domain 중 max length로 맞춰야 함
        svid_df['STEP01-25_sensors'] = svid_df['STEP01-25_sensors'].apply(lambda x: get_numpy_from_nonfixed_2d_array(x, fixed_length, padding_value=0))
        return svid_df
    
    def get_X_y(self, df):
        X = np.expand_dims(df['STEP01-25_sensors'].values[0], axis=0)
        for arr in tqdm(df['STEP01-25_sensors'].values[1:]):
            X = np.vstack([X, np.expand_dims(arr, axis=0)]).astype(np.float32)
        X = np.swapaxes(X,1,2)
        y = df['fault'].values.astype(np.float32)
        return X, y