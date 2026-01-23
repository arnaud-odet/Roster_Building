import pandas as pd
import numpy as np
import os
from pathlib import Path
import copy

from roster_lib.constants import RAW_DATA_PATH, PREPROC_DATA_PATH
from roster_lib.id_dict import pid2pos_bref

POSITION_MAPPING = {'PG':1, 'SG':2, 'SF':3, 'PF':4, 'C':5}

class Loader :
    
    def __init__(self):
        self.raw_data_path = RAW_DATA_PATH / 'muniz_data'
        self.preproc_data_filepath = PREPROC_DATA_PATH / 'clustering' / 'concatenated_data.csv'
        
        if os.path.exists(self.preproc_data_filepath):
            self._load_preproc_data()
        else :
            self._preprocess_data()        
    
    def _load_preproc_data(self):
        self.df = pd.read_csv(self.preproc_data_filepath, index_col = 0)    
    
    def _preprocess_data(self):
        self._load_raw_data()
        self._handle_duplicated()
        self.df = self._clean_data()
        self.df.to_csv(self.preproc_data_filepath)
    
    def _load_raw_data(self):   
        self.raw_data = {f[6:-4] : pd.read_csv(self.raw_data_path / f, index_col=0).set_index('pidSzn') for f in os.listdir(self.raw_data_path) if 'Clutch' not in f and f[-3:]=='csv'}
        
    def _handle_duplicated(self):
        rest_data = copy.deepcopy(self.raw_data)
        for k,v in rest_data.items():
            issues = v[v.index.duplicated()].index.unique()
            v['discard'] = [id in issues for id in v.index] 
            v.reset_index(inplace=True)
            for id in issues:
                keep_index = v[v['pidSzn']==id].sort_values('GP', ascending=False).index[0]
                v.loc[keep_index,'discard'] = False
            v.set_index('pidSzn', inplace= True)
            rest_data[k] = v[v['discard']==False].drop(columns = 'discard')
        self.preproc_data = rest_data
        
    def _merge_data(self):
        df = pd.concat([v.drop(columns = ['MIN', 'GP', 'Season']) for v in self.preproc_data.values()], axis = 1)
        return df
    
    def _clean_data(self):
        df = self._merge_data()
        # Columns with most NA is FG3_PCT as some player do not shoot 3s. Filling with 0s
        df['FG3_PCT'] = df['FG3_PCT'].fillna(0)
        df['FG2_PCT'] = df['FG2_PCT'].fillna(0)
        df['position'] = df.index.map(pid2pos_bref).map(POSITION_MAPPING)
        # 32 players are partly missing, none of them playing more than 10 minutes in any dataset, droping them
        df = df.dropna()
        return df
    
