import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

from roster_lib.preprocessing.loader import Loader
from roster_lib.constants import PREPROC_DATA_PATH

MANUAL_DROP_COLS = {'Score': ['TS_PCT'],
             'Misc': ['DIST_MILES', 'AVG_SPEED', 'LOOSE_BALLS_RECOVERED'],
             'Defense': ['FREQ','D_FGM','D_FGA'],
             'Pass': ['TOUCHES','TOV','PASSES_MADE','AST'],
             'Rebound': ['BOX_OUTS']}

MANUAL_INCLUSION_COLS = {
    'Score': [
        'MIN', 
        'GP',
        'EFG_PCT',
        'USG_PCT',
        # 'PTS_OFF_TOV',
        'PTS_2ND_CHANCE',
        # 'PTS_FB',
        'PTS_PAINT',
        'DRIVE_PTS',
        # 'DRIVE_FG_PCT',
        # 'CATCH_SHOOT_PTS',
        # 'CATCH_SHOOT_FG_PCT',
        # 'PULL_UP_PTS',
        # 'PULL_UP_FG_PCT',
        # 'PAINT_TOUCH_PTS',
        # 'PAINT_TOUCH_FG_PCT',
        # 'POST_TOUCH_PTS',
        # 'POST_TOUCH_FG_PCT',
        # 'ELBOW_TOUCH_PTS',
        # 'ELBOW_TOUCH_FG_PCT',
        'FG2A',
        'FG2_PCT',
        'FG3A',
        'FG3_PCT',
        ],
    'Misc': [
        'OFF_LOOSE_BALLS_RECOVERED',
        'DEF_LOOSE_BALLS_RECOVERED',
        'PFD',
        # 'DRIVE_PF',
        ],
    'Defense': [
        'STL',
        'BLK',
        # 'DEF_RIM_FGA',
        # 'DEF_RIM_FG_PCT',
        # 'D_FGA',
        # 'D_FG_PCT',
        'PCT_PLUSMINUS',
        # 'OPP_PTS_2ND_CHANCE',
        # 'OPP_PTS_FB',
        # 'OPP_PTS_PAINT',
        'DEFLECTIONS',
        'CHARGES_DRAWN',
        # 'CONTESTED_SHOTS_2PT',
        # 'CONTESTED_SHOTS_3PT',
        ],
    'Pass': [
        'SCREEN_ASSISTS',
        # 'PAINT_TOUCH_PASSES',
        # 'PAINT_TOUCH_AST',
        # 'PAINT_TOUCH_TOV',
        # 'PAINT_TOUCHES',
        # 'POST_TOUCH_PASSES',
        # 'POST_TOUCH_AST',
        # 'POST_TOUCH_TOV',
        # 'POST_TOUCHES',
        # 'ELBOW_TOUCH_PASSES',
        # 'ELBOW_TOUCH_AST',
        # 'ELBOW_TOUCH_TOV',
        # 'ELBOW_TOUCHES',
        # 'FT_AST',
        # 'SECONDARY_AST',
        # 'POTENTIAL_AST',
        # 'AST_TO_PASS_PCT',
        # 'DRIVE_PASSES',
        # 'DRIVE_AST',
        # 'DRIVE_TOV',
        # 'FRONT_CT_TOUCHES',
        'TIME_OF_POSS',
        # 'AVG_SEC_PER_TOUCH',
        'AVG_DRIB_PER_TOUCH',
        'AST_PCT',
        'AST_TO',
        'AST_RATIO'
        ],
    'Rebound': [
        'OREB_UNCONTEST',
        # 'OREB_CONTEST_PCT',
        # 'OREB_CHANCES',
        # 'OREB_CHANCE_PCT',
        # 'OREB_CHANCE_DEFER',
        'DREB_CONTEST',
        'DREB_UNCONTEST',
        # 'DREB_CHANCES',
        # 'DREB_CHANCE_PCT',
        # 'DREB_CHANCE_DEFER',
        # 'AVG_OREB_DIST',
        # 'AVG_DREB_DIST',
        'OFF_BOXOUTS',
        'DEF_BOXOUTS',
        # 'BOX_OUT_PLAYER_TEAM_REBS',
        # 'BOX_OUT_PLAYER_REBS',
        # 'PCT_BOX_OUTS_OFF',
        # 'PCT_BOX_OUTS_DEF',
        # 'PCT_BOX_OUTS_TEAM_REB',
        # 'PCT_BOX_OUTS_REB'
        ]
    }

class FeatureHandler :
    
    def __init__(self, use_positions:bool=True, feature_version:int = None, verbose :bool = True):
        loader = Loader()
        self.verbose = verbose
        if not hasattr(loader, 'preproc_data'):            
            loader._load_raw_data()
            loader._handle_duplicated()
        self.loader = loader
        self.df = loader.df
        self.feature_version = feature_version
        self._create_features_dict()
        if not use_positions:
            self.df.drop(columns = 'position', inplace=True)
        else :
            self.incl.append('position')        
        
    def _create_features_dict(self):
        if self.feature_version == None :
            auto_drop_cols = {k : self.auto_excl_vif( v.drop(columns = ['MIN','GP','Season']).dropna()) for k, v in self.loader.preproc_data.items()}
            self.autoexcl_dict = auto_drop_cols
            self.excl_dict = MANUAL_DROP_COLS
            self.incl_dict = MANUAL_INCLUSION_COLS
        else :
            feature_filepath = PREPROC_DATA_PATH / 'clustering' / f'features_v{self.feature_version}.json'
            with open(feature_filepath, "r") as f:
                features = json.load(f)
            self.autoexcl_dict = features['autoexcl']
            self.excl_dict = features['excl']
            self.incl_dict = features['incl']
    
        self.autoexcl = []
        for v in self.autoexcl_dict.values():
            self.autoexcl += v
        self.excl = []
        for v in self.excl_dict.values():
            self.excl += v
        self.incl = []
        for v in self.incl_dict.values():
            self.incl += v

    def get_data(self, feature_selection:str= None):
        if feature_selection == None:
            return self.df
        return self.df[self.incl] if feature_selection == 'incl' else self.df.drop(columns = self.excl if feature_selection =='excl' else self.autoexcl)

    
    def auto_excl_vif(self, df, threshold:float=10):
        _df = df.copy()
        if 'FG3_PCT' in _df.columns :
            _df['FG3_PCT'] = _df['FG3_PCT'].fillna(0)
        if 'FG2_PCT' in _df.columns :
            _df['FG2_PCT'] = _df['FG2_PCT'].fillna(0)
        keep_iterating = True
        drop_cols = []
        if self.verbose:
            print(f"Initial number of features : {_df.shape[1]}")
        while keep_iterating :
            vif_df = pd.DataFrame()
            vif_df["feature"] = _df.columns
            vif_df["vif_index"] = [vif(_df.values, i) for i in range(_df.shape[1])]
            vif_df.sort_values(by="vif_index", ascending=False, inplace=True)
            top_vif = vif_df.iloc[0]['vif_index']
            top_vif_ft = vif_df.iloc[0]['feature']
            msg = f"Top VIF = {top_vif:.1f}"
            if top_vif > threshold :
                _df = _df.drop(columns = top_vif_ft)
                drop_cols.append(top_vif_ft)
                msg += f" - droping feature {top_vif_ft}"
            else :
                keep_iterating = False
                msg += f" - stopping features exclusion with {_df.shape[1]} feature retained"
            if self.verbose: 
                print(msg)
            
        return drop_cols    
    
    def compute_vifs(self):
        self.vifs = {}
        for fs in ['incl', 'excl', 'autoexcl']:
            df = self.get_data(feature_selection=fs)
            vif_df = pd.DataFrame()
            vif_df["features"] = df.columns
            vif_df["vif_index"] = [vif(df.values, i) for i in range(df.shape[1])]
            self.vifs[fs] = vif_df.sort_values(by="vif_index", ascending=False)