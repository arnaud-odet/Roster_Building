import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os

RAW_DATA_PATH = Path('./raw_data')
PREPROC_DATA_PATH = Path('./preproc_data')

# Data Loading, concatenating and feature engineering

def load_raw_data(sources:list):
    data = {}
    for source in sources :
        s_path = RAW_DATA_PATH / source
        dfs = [pd.read_excel(s_path / f, engine = 'openpyxl', index_col=0) for f in os.listdir(s_path)]
        data[source] = pd.concat(dfs)
        data[source] = data[source].rename(columns = {'Player':'PLAYER'}).set_index('PLAYER')
    return data

def concatenate_data(data):
    col_list = []
    for i, (k, df) in enumerate(data.items()):
        for col in df.columns :
            if col in col_list:
                df.drop(columns = col, inplace = True)
        if i == 0 :
            conc_df = df.copy()
        else :
            conc_df = conc_df.merge(df, left_index = True, right_index = True)
        col_list += df.columns.tolist()
    return conc_df

def convert_to_int(nb_str):
    return int(nb_str.replace(',',''))

def feature_engineering(df, time_norm:bool = False):
    odf = df.rename(columns = {col_name : col_name.replace('\xa0', '_') for col_name in df.columns}).copy()
    features_to_drop = ['AGE', 'W', 'L', 'TIME_RATIO']
    features_w_volume = ['PTS', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'FP', 'DD2', 'TD3', 'PIE', 'POSS', 'OPP_PTSOFF_TOV', 'OPP_PTS2ND_CHANCE', 'OPP_PTSFB', 'OPP_PTSPAINT']
    
    odf['TOT_MIN'] = odf['GP'] * odf['MIN']
    odf["FT/FGA"] = odf['FTA'] / odf['FGA']
    odf["FT/FGA"] = odf["FT/FGA"].fillna(0)
    odf['POSS'] = odf['POSS'].apply(convert_to_int)
    if time_norm :
        odf['TIME_RATIO'] = odf['GP'] / odf['TOT_MIN']
        for ft in features_w_volume :
            odf[ft] = odf[ft] * odf['TIME_RATIO']
    odf['player'] = [ind.split('_')[-1] for ind in odf.index]
    odf['year'] = [ind.split('_')[1] for ind in odf.index]
    odf.index.name = 'id'
    return odf.drop(columns = [ft for ft in features_to_drop if ft in odf.columns])
    
    
# Pipeline

def preprocessing(time_norm:bool = False):
    
    conc_data_filepath = PREPROC_DATA_PATH / f"concatenated_data{'_timenorm' if time_norm else ''}.csv"
    
    if os.path.exists(conc_data_filepath):
        df = pd.read_csv(conc_data_filepath, index_col=0)
    else :
        data = load_raw_data(sources = ['per_game_data','advanced_data','defense_data'])
        df = concatenate_data(data)
        df = feature_engineering(df, time_norm=time_norm)
        df.to_csv(conc_data_filepath)
    
    return df
