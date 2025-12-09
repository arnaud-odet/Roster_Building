import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def jaccard_similarity(a:list,b:list):
    numerator = 0
    for item in a :
        if item in b :
            numerator +=1
    c = a + b
    denominator = len(set(c))
    return numerator / denominator

def compute_transition(cluster_df: pd.DataFrame, roster_df: pd.DataFrame, team: str, year: int):
    prev_yr = year - 1
    prev_yr_roster = roster_df[(roster_df['YEAR'] == prev_yr) & (roster_df['TEAM'] == team)]['PLAYER_NAME'].unique().tolist()
    curr_yr_roster = roster_df[(roster_df['YEAR'] == year) & (roster_df['TEAM'] == team)]['PLAYER_NAME'].unique().tolist()
    jacc = jaccard_similarity(prev_yr_roster, curr_yr_roster)
    
    cl_cols = cluster_df.drop(columns = ['YEAR','TEAM','tot_min_season', 'player_jaccard', 'player_cosine', 'cluster_cosine']).columns
    prev_yr_state = cluster_df[(cluster_df['YEAR'] == prev_yr) & (cluster_df['TEAM'] == team)][cl_cols]
    curr_yr_state = cluster_df[(cluster_df['YEAR'] == year) & (cluster_df['TEAM'] == team)][cl_cols]
    clust_cos_sim = cosine_similarity(prev_yr_state, curr_yr_state)
    
    return clust_cos_sim[0][0], jacc

def transition_matrix(cluster_df:pd.DataFrame, roster_df:pd.DataFrame, debug_verbose:bool = False ) :
    
    tr_df = cluster_df.copy()
    tr_df['player_jaccard'] = np.nan
    tr_df['player_cosine'] = np.nan    
    tr_df['cluster_cosine'] = np.nan
    
    for tm in tr_df['TEAM'].unique() :
        playtime_df = roster_df[roster_df['TEAM']==tm].pivot(index = 'PLAYER_NAME', columns = 'YEAR', values = 'TOT_MIN').fillna(0)
        playtime_df = playtime_df / playtime_df.sum(axis = 0)
        for i, yr in enumerate(tr_df[tr_df['TEAM'] == tm].sort_values('YEAR')['YEAR'].unique()):
            if i > 0 :
                if debug_verbose:
                    print(tm, yr, end='\r')
                ind = tr_df[(tr_df['TEAM']==tm) & (tr_df['YEAR']==yr)].index
                cl_cs, jacc = compute_transition(cluster_df=tr_df, roster_df=roster_df, team = tm, year = yr)
                pl_cs = cosine_similarity(playtime_df[[yr]].T, playtime_df[[yr-1]].T)[0][0]
                tr_df.loc[ind,'player_jaccard'] = jacc
                tr_df.loc[ind,'player_cosine'] = pl_cs
                tr_df.loc[ind,'cluster_cosine'] = cl_cs
                if debug_verbose:
                    print(tm, yr, np.round(jacc,2), np.round(pl_cs,2), np.round(cl_cs,2), end='\n')
    
    return tr_df


