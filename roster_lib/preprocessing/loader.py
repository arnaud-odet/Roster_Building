import pandas as pd
import numpy as np
import os
from pathlib import Path
import copy

from roster_lib.constants import RAW_DATA_PATH, PREPROC_DATA_PATH
from roster_lib.id_dict import pid2pos_bref

POSITION_MAPPING = {'PG':1, 'SG':2, 'SF':3, 'PF':4, 'C':5}
VOLUME_FEATURES = [
 'DD2',
 'TD3',
 'PTS_OFF_TOV',
 'PTS_2ND_CHANCE',
 'PTS_FB',
 'PTS_PAINT',
 'PTS_PER_ELBOW_TOUCH',
 'PTS_PER_POST_TOUCH',
 'PTS_PER_PAINT_TOUCH',
 'DRIVE_PTS',
 'CATCH_SHOOT_PTS',
 'PULL_UP_PTS',
 'PAINT_TOUCH_PTS',
 'POST_TOUCH_PTS',
 'ELBOW_TOUCH_PTS',
 'FG2A',
 'FG3A',
 'OFF_LOOSE_BALLS_RECOVERED',
 'DEF_LOOSE_BALLS_RECOVERED',
 'LOOSE_BALLS_RECOVERED',
 'PFD',
 'DRIVE_PF',
 'DIST_MILES',
 'DIST_MILES_OFF',
 'DIST_MILES_DEF',
 'STL',
 'BLK',
 'DEF_RIM_FGM',
 'DEF_RIM_FGA',
 'D_FGM',
 'D_FGA',
 'OPP_PTS_2ND_CHANCE',
 'OPP_PTS_FB',
 'OPP_PTS_PAINT',
 'DEFLECTIONS',
 'CONTESTED_SHOTS_2PT',
 'CONTESTED_SHOTS_3PT',
 'SCREEN_ASSISTS',
 'SCREEN_AST_PTS',
 'PAINT_TOUCH_PASSES',
 'PAINT_TOUCHES',
 'POST_TOUCH_PASSES',
 'POST_TOUCH_AST',
 'POST_TOUCH_TOV',
 'POST_TOUCHES',
 'ELBOW_TOUCH_PASSES',
 'ELBOW_TOUCH_AST',
 'ELBOW_TOUCHES',
 'AST',
 'FT_AST',
 'SECONDARY_AST',
 'POTENTIAL_AST',
 'PASSES_MADE',
 'PASSES_RECEIVED',
 'DRIVE_PASSES',
 'DRIVE_AST',
 'DRIVE_TOV',
 'TOUCHES',
 'FRONT_CT_TOUCHES',
 'TIME_OF_POSS',
 'AST_RATIO',
 'TOV',
 'OREB_UNCONTEST',
 'OREB_CHANCES',
 'DREB_CONTEST',
 'DREB_UNCONTEST',
 'DREB_CHANCES',
 'DREB_CHANCE_DEFER',
 'OFF_BOXOUTS',
 'DEF_BOXOUTS',
 'BOX_OUTS',
 'BOX_OUT_PLAYER_TEAM_REBS',
 'BOX_OUT_PLAYER_REBS',
]

class Loader :
    
    def __init__(self, time_norm: bool = True):
        self.raw_data_path = RAW_DATA_PATH / 'muniz_data'
        self.time_norm = time_norm
        tn_str = 'time_norm' if time_norm else 'raw'
        self.preproc_data_filepath = PREPROC_DATA_PATH / 'clustering' / f'concatenated_data_{tn_str}.csv'
        
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
        if self.time_norm:
            for col in VOLUME_FEATURES:
                self.df[col] =  self.df[col] / self.df['MIN']
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
        df = df.merge(self.preproc_data['Score'][['MIN']], left_index=True, right_index=True, how = 'left')
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
    
