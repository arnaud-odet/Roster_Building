import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import pickle
import copy
import itertools
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import hmean
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from roster_lib.data.season import Season
from roster_lib.utils.transition import jaccard_similarity
from roster_lib.utils.plots import plot_logistic_decision_boundary_2d

RAW_DATA_PATH = Path("/home/admin/code/arnaud-odet/7_PhD/Roster/raw_data")
PREPROC_DATA_PATH = Path("/home/admin/code/arnaud-odet/7_PhD/Roster/preproc_data")


class Period:
    
    def __init__(self, start:int, end:int, non_dependence_k:int = 5):
        self.period = range(start, end +1)
        self.lineups_path = PREPROC_DATA_PATH / 'period' / f'{start}_{end}_consolidated_lineups.csv'
        self.success_path = PREPROC_DATA_PATH / 'period' / f'{start}_{end}_consolidated_success.csv'
        self.rosters_path = PREPROC_DATA_PATH / 'period' / f'{start}_{end}_consolidated_rosters.pkl'
        
        self._build_arch()
        self._load_consolidated_data()
        self._preprocess_consolidated_data()
        self.success.set_index(['team','year'], inplace = True)
        self.transitions = self._compute_transitions()
        self.ordered_rosters = self._order_rosters()
        self._add_ranks()
        self._add_diversity_index()
        self._add_non_dependence_score(top_k = non_dependence_k)
        lu_characteristics_cols = [col for col in self.success.columns if not col in self.transitions.columns]
        self.transitions = self.transitions.merge(self.success[lu_characteristics_cols], left_index = True, right_index = True)
        self.copresences = self._copresence_matrix()
        self._check_copresences()
        
    # Hidden  
          
    def _build_arch(self):
        for filepath in [self.lineups_path, self.success_path, self.rosters_path]:
            if not os.path.exists(filepath.parent) :
                os.makedirs(filepath.parent)        
        
    def _load_consolidated_data(self):
        if os.path.exists(self.lineups_path) and os.path.exists(self.success_path) and os.path.exists(self.rosters_path):
            self.data = pd.read_csv(self.lineups_path, index_col=0)
            self.success = pd.read_csv(self.success_path, index_col=0)
            with open(self.rosters_path, 'rb') as f:
                self.rosters = pickle.load(f)
        else :
            self.data, self.success, self.rosters = self._consolidate_data()
    
    def _preprocess_consolidated_data(self):
        _tdf = self.data.groupby(['year','team']).agg({'time':'sum'}).rename(columns = {'time':'tot_time'}).reset_index()
        _tdf['id'] = [f"{yr}_{tm}" for yr, tm in zip(_tdf['year'], _tdf['team'])]
        _ts = _tdf.set_index('id')['tot_time']
        self.data['tmp_id'] = [f"{yr}_{tm}" for yr, tm in zip(self.data['year'], self.data['team'])]
        self.data = self.data.merge(_ts, left_on = 'tmp_id', right_index = True)
        self.data['relative_time'] = self.data['time'] / self.data['tot_time']
        self.data.drop(columns = ['tmp_id','tot_time'], inplace=True)
        
    def _consolidate_data(self):
        _tmp_lu = []
        rosters = {}
        for yr in self.period:
            print(f"------> PROCESSING SEASON {yr}-{yr+1} <------")
            season = Season(yr, keep_playoffs=False) # Discarding playoff game for fair diversity index
            success = season.success
            rosters[yr] = season.rosters
            for team, df in season.team_lineup_stats.items():
                _tdf = df.reset_index()[['id','lineup','time','pm','pm_per_48min']].copy()
                _tdf['team'] = team
                _tdf['year'] = yr
                _tdf['win_rate'] = success.loc[team,'win_rate']
                _tdf['csf'] = success.loc[team,'csf']
                _tmp_lu.append(_tdf)

        ludf = pd.concat(_tmp_lu).reset_index(drop=True)
        for k in range(5):
            ludf[k] = [item.split('_')[k] for item in ludf['lineup']]
            
        # Save files
        ludf.to_csv(self.lineups_path)
        sdf = ludf[['team','year','win_rate','csf']].drop_duplicates()
        sdf.to_csv(self.success_path)
        with open(self.rosters_path, 'wb') as f:
            pickle.dump(rosters, f, protocol= pickle.HIGHEST_PROTOCOL)
        
        return ludf, sdf, rosters    
    
    def _compute_transitions(self):
        ts = []
        for tm in self.data['team'].unique():
            tdf = self.data[self.data['team']==tm].pivot(index = 'lineup', columns = 'year', values = 'relative_time').fillna(0)
            for i, yr in enumerate(self.period) :
                if i > 0 :
                    prev_yr = tdf[yr-1].values.reshape(1,-1)
                    curr_yr = tdf[yr].values.reshape(1,-1)
                    delta_wr = self.success.loc[(tm,yr),'win_rate'] - self.success.loc[(tm,yr-1),'win_rate']
                    prev_yr_roster = self.rosters[yr-1][tm]
                    curr_yr_roster = self.rosters[yr][tm]
                    transition_jacc = jaccard_similarity(list(prev_yr_roster.keys()), list(curr_yr_roster.keys()))
                    _tmp_pl_df = pd.DataFrame.from_dict(prev_yr_roster, orient='index').rename(columns = {0:yr-1})\
                        .merge(pd.DataFrame.from_dict(curr_yr_roster, orient='index').rename(columns = {0:yr}), 
                               left_index = True, right_index = True, how = 'outer').fillna(0)
                    prev_yr_pl = _tmp_pl_df[yr-1].values.reshape(1,-1)
                    curr_yr_pl = _tmp_pl_df[yr].values.reshape(1,-1)
                    ts.append({'team':tm, 
                            'year':yr, 
                            'lu_cosine': cosine_similarity(prev_yr, curr_yr)[0][0],
                            'pl_cosine' : cosine_similarity(prev_yr_pl, curr_yr_pl)[0][0],
                            'jaccard' : transition_jacc,
                            'win_rate' : self.success.loc[(tm,yr),'win_rate'],
                            'delta_win_rate': delta_wr,
                            'csf': self.success.loc[(tm,yr),'csf']})
        return pd.DataFrame(ts).set_index(['team','year'])
    
    def _order_rosters(self):
        _tmp_rosters = copy.deepcopy(self.rosters)
        _tmp_rosters = {k : {vk : {vv : i+1 for i, vv in enumerate(vv.keys())} for vk, vv in v.items()} for k,v in _tmp_rosters.items()}
        ordered_rosters = {}
        for k,v in _tmp_rosters.items():
            for vk,vv in v.items() :
                ordered_rosters[f'{vk}_{k}'] = vv
        return ordered_rosters

    def _add_ranks(self):
        for i in range(5):
            self.data[f'rk_{i}'] = ''
        rks = []
        for index, row in self.data.iterrows():
            key = row['id'][:8]
            rks.append(sorted([ self.ordered_rosters[key][row[f'{i}']] for i in range(5)]))
        self.data[[f'rk_{i}' for i in range(5)]] = rks
        self.data['rk_id'] = self.data.apply(lambda row : f"{row['rk_0']}_{row['rk_1']}_{row['rk_2']}_{row['rk_3']}_{row['rk_4']}", axis = 1)
        self.data['use_id'] = [txt[-3:] for txt in self.data['id']]
        self.data['use_id'] = self.data['use_id'].astype(int)
   
    def _add_diversity_index(self): #A REVOIR 
        _df = self.data[['team','year','id','time','relative_time']].copy()
        _tmp = pd.concat([_df[_df['relative_time'] > i/100].groupby(['team','year']).count()[['id']].rename(columns ={'id':i})  for i in range(1,10)], axis=1)
        self.success['diversity_index'] = 0
        for (tm,yr), row in _tmp.iterrows():
            _ = (row >= _tmp.columns).astype(int) * _tmp.columns
            diversity_index = np.argmax(_) +1
            self.success.loc[(tm,yr),'diversity_index'] = diversity_index
  
    def _add_non_dependence_score(self, top_k:int = 5):
        _ndpd = [self.top_players_lineups_rate(top_k = i+1, exclude = True).rename(columns = {'time_share':f'wo_{i+1}'}) for i in range(top_k)]
        _ndpd_df = pd.concat(_ndpd, axis = 1)
        _ndpd_df['nondpd_score'] = [hmean(row) for index,row in _ndpd_df.iterrows()] 
        self.success = self.success.merge(_ndpd_df['nondpd_score'], left_index = True, right_index= True)

    def _copresence_matrix(self):
        # Helper functions
        def _identify_copresence(str_id, id_a, id_b):
            pl_list = str_id.split('_')
            return f'{id_a}' in pl_list and f'{id_b}' in pl_list 

        def _compute_copresence_rate(df, id_a, id_b):
            df['copresence'] = df['rk_id'].apply(lambda x : _identify_copresence(x, id_a, id_b))
            df['wt_copr'] = df['copresence'].astype(int) * df['relative_time']
            return df['wt_copr'].sum()

        # Main function
        _df = self.data.copy()
        copresences = {}
        for year in _df['year'].unique():
            for team in _df['team'].unique():
                str_id = f"{team}_{year}"
                _subdf = _df[(_df['team']==team)&(_df['year']==year)].copy()
                max_pl = _subdf['rk_4'].max()
                copr_df = pd.DataFrame([[ _compute_copresence_rate(_subdf, i, j) for i in range(1, max_pl + 1)] for j in range(1, max_pl + 1)], index = range(1, max_pl + 1), columns=range(1, max_pl + 1))
                copresences[str_id] = copr_df
        return copresences

    def _check_copresences(self):
        err_list = []
        for (tm,yr) in self.success.index:
            
            str_id = f'{tm}_{yr}'
            playtime_ratios = [v / (sum(self.rosters[yr][tm].values()) / 5) for v in self.rosters[yr][tm].values()]
            check = np.allclose(playtime_ratios, [self.copresences[str_id].loc[k,k] for k in self.copresences[str_id].index])
            if not check :
                err_list.append((tm, yr))
        if len(err_list) != 0:
            print(f"Error in corpresences matrices for games {err_list}")
        
    # Visible

    def top_players_lineups_rate(self, 
                                top_k:int = 5, 
                                exclude:bool = False):
        """
        Returns a dataframe indicating the percentage of play time with a lineup :
            - made up only by players from the top_k is playing if exclude = False and top_k >=5,
            - including all top_k players if exclude = False and top_k <5,
            - excluding all players from the top_k if playing if exclude = True.
        """
        max_n_pl = max([len(v) for v in self.ordered_rosters.values()])
        df = self.data.copy()
        if top_k < 5 and not exclude :
            strid = "_".join([f'{k+1}' for k in range(top_k)])
            df['focus_lineups'] = [rkid[:len(strid)] == strid for rkid in df['rk_id']]
        else :
            focus_indices = [str(i) for i in range(top_k +1 , max_n_pl +1)] if exclude else [str(i) for i in range(1, top_k +1)]
            focus_lineups = ["_".join(ll) for ll in itertools.combinations(focus_indices,5)] 
            df['focus_lineups'] = df['rk_id'].isin(focus_lineups)
        tot_time = df.pivot_table(values= 'time', index = 'team', columns = 'year', aggfunc='sum')
        top_time = df[df['focus_lineups']].pivot_table(values= 'time', index = 'team', columns = 'year', aggfunc='sum')
        return (top_time / tot_time).fillna(0).unstack().to_frame().rename(columns = {0:'time_share'})        

    def diversity_plot(self, 
                       threshold : float = 0.02,
                       plot_boundaries:bool=True, 
                       ax = None, 
                       show_confusion_matrices:bool = True, 
                       show_summary:bool = True,
                       step:float=0.05):
        _df = self.data.copy()
        _df['considered'] = (_df['relative_time'] > threshold).astype(int)
        _df = _df.groupby(['year','team']).agg({'lineup':'count','considered':'sum'})
        _df = _df.merge(self.success, left_index = True, right_index = True)
        
        plot_logistic_decision_boundary_2d(data = _df,
                                           x1 = 'lineup',
                                           x2 = 'considered',
                                           y = 'csf',
                                           scale = True,
                                           hue_col='win_rate',
                                           step = step,
                                           plot_boundaries=plot_boundaries,
                                           show_confusion_matrices=show_confusion_matrices,
                                           show_summary=show_summary,
                                           ax=ax)
        
    def illustrate_lineups(self, k:int = 4, normalize:bool = False, normalization_threshold:int = 48):
        """
        Displays the lineups repartition of :
            - the top k teams of the period in terms of win rate
            - k random teams of the period, not part of the top nor the bottom,
            - the bottom k teams of the period
        in terms of time played (x-axis, in minutes) and +/- (y-axis, in points).
        If normalize is set to True, all +/- of lineups having played more than 12 minutes are displayed in "+/- per 48 min"
        """
        ludf = self.data.copy()
        
        ludf['tm_yr'] = [strid[:8] for strid in ludf['id']]
        ludf['time'] = ludf['time'] / 60
        if normalize :
            above_threshold = (ludf['time'] > normalization_threshold).astype(int)
            ludf['pm_norm'] = above_threshold * ludf['pm_per_48min'] + (1 - above_threshold) * ludf['pm']
        top_teams = [f'{tm}_{yr}' for tm, yr in self.success.sort_values('win_rate',ascending=False).iloc[:k].index]
        random_teams = [ ludf.sort_values('win_rate',ascending=False).iloc[i]['tm_yr'] for i in random.sample(range(10, ludf.shape[0]-10),k)]
        bottom_teams = [f'{tm}_{yr}' for tm, yr in self.success.sort_values('win_rate',ascending=False).iloc[-k:].index]
        fig, axs = plt.subplots(3,k,figsize = (5*k,15))
        gt = 1
        var_y = 'pm_norm' if normalize else 'pm'
        for i,ax in enumerate(axs.flatten()):
            focus = [top_teams,random_teams,bottom_teams][i//k][i%k]
            sns.scatterplot(data = ludf[ludf['tm_yr'] == focus], x = 'time', y = var_y, ax = ax);
            ax.set_title(f"{focus} - WinRate = {self.success.loc[(focus[:3], int(focus[-4:])),'win_rate']:.3f}, CSF = {bool(self.success.loc[(focus[:3], int(focus[-4:])),'csf'])}");
            ax.plot(ax.get_xlim(), [0,0]);
            ax.plot([normalization_threshold,normalization_threshold], ax.get_ylim());
            ax.set_xlabel("Time (in minutes)");
            ax.set_ylabel(f"+/- {'per 48 minutes' if normalize else ''}")

    def brackets(self, time_separators:list=[24,48,96], pm_separators:list=[0]):
        """
        Return the count of lineups per time brackets (in minutes) and +/- brackets
        where brackets are determined by the separators specified by the user.
        """
        def _return_bracket(x, list):
            for i, (low, high) in enumerate(zip(list, list[1:])):
                if low <= x <= high :
                    return f"{low}_{high}"
                    
        time_separators.sort()
        pm_separators.sort()
        time_brackets = [0] + time_separators + [np.inf]
        pm_brackets = [-np.inf] + pm_separators + [np.inf]
        
        _df = self.data.copy()
        _df['time'] = _df['time'] / 60
        _df['time_bracket'] = _df['time'].apply(lambda x : _return_bracket(x, time_brackets))
        _df['pm_bracket'] = _df['pm'].apply(lambda x : _return_bracket(x, pm_brackets))
        
        bracket_df = _df.pivot_table(index =['team','year'], columns = ['time_bracket','pm_bracket'], values = 'lineup', aggfunc = 'count').fillna(0).astype(int)
        bracket_df.columns = ['t:[' + ']/pm:['.join(col).strip() +']' for col in bracket_df.columns.values]
        
        return bracket_df
    
    def find_non_copresence(self, top_k:int, copresence_threshold:float = 0.001):
        """
        Returns a dictionnary of teams in which two players of the top k have no co-presence :
            - Keys : a string team_year
            - Values : the list of players with no copresences with rank_id and name
        Args :
            - top_k : the rank of the players of which the non-copresence is investigating
            - copresence_threshold : a float indicating the percentage of the total team playtime 
        """
        results = {}
        for k,v in self.copresences.items():
            _df = v <= copresence_threshold
            non_copr = _df.where(_df).stack().index.tolist()
            focus_non_copr = [pair for pair in non_copr if pair[0] <= top_k and pair[1] <= top_k and pair[0]<pair[1]]
            if len(focus_non_copr) != 0 :
                ordered_roster = copy.deepcopy(self.ordered_rosters[k])
                inversed_roster = {v : k for k,v in ordered_roster.items()}
                results[k] = {'rank_id' : focus_non_copr,
                            'names' : [(inversed_roster[a].split(' - ')[0], inversed_roster[b].split(' - ')[0]) for (a,b) in focus_non_copr]}
        return results