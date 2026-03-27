import pandas as pd
import numpy as np
from itertools import combinations_with_replacement, product
from scipy.stats import ttest_ind

class LineupExplorer:
    
    def __init__(self, ludf, success_df, ignore_cluster_0:bool = True):
        self.df = ludf
        self.success_df = success_df
        self.clusters = [int(col.split('_')[-1]) for col in ludf.columns if col[:8] == 'nb_pl_cl']
        if 0 in self.clusters and ignore_cluster_0:
            self.clusters.remove(0)
        
    def _create_base_sample_df(self, compo:dict={}):
        if compo == {}:
            return self.df
        else :
            _masks = np.zeros((self.df.shape[0], len(compo)))
            for i, (k,v) in enumerate(compo.items()):
                _count_col = f'nb_pl_cl_{k}'
                _mask = self.df[_count_col] >= v
                _masks[:,i] = _mask
            _final_mask = _masks.all(axis = 1)
            return self.df[_final_mask].copy()
                
    def _sample_distribution(self, sample_df ,target:str = 'pm_per_48min', weight:bool = True):
        _valid_target = ['win_rate', 'csf', 'pm_per_48min', 'pm']
        if not target in _valid_target :
            raise ValueError(f"target must be in {_valid_target}, got '{target}' instead")
        else :                    
            if target in ['win_rate','csf']:
                _law_df = sample_df[['team','year','time', 'pm']].groupby(['team','year']).sum().merge(self.success_df[['win_rate','csf']],right_index = True, left_index = True)
            else  :
                _law_df = sample_df.copy()
            _law_df['weight'] = (_law_df['time'] / _law_df['time'].sum()) if weight else 1 / _law_df.shape[0]
            return _law_df[['weight', target]]
        
    def _draw_from_distribution(self, law_df, size:int = 1000):
        if law_df.shape[0] == 0:
            return np.zeros(size)
        _law = law_df.values
        return np.random.choice(a = _law[:,1], size= size, replace = True, p = _law[:,0])
    
    def _create_samples_data(self, new_addition:int, base_compo:dict = {}):
        _sample_df = self._create_base_sample_df(compo = base_compo)
        _threshod = base_compo.get(new_addition,0) 
        _mask = _sample_df[f'nb_pl_cl_{new_addition}'] > _threshod
        _sample_with = _sample_df[_mask]
        _sample_without = _sample_df[_mask == False]
        return _sample_with, _sample_without
    
    def explore_tree(self, compo:dict = {}):
        _df = self._create_base_sample_df(compo)
        _tot_time = _df['time'].sum()
        _tree_df = _df[[f'nb_pl_cl_{i}' for i in range(1,11)]].copy()
        for k,v in compo.items():
            _tree_df[f'nb_pl_cl_{k}'] -= v
        tree_leaves = {}
        for k in range(1,11):
            if _tree_df[f'nb_pl_cl_{k}'].sum() > 0 :
                tree_leaves[k] = float(_df[_df[f'nb_pl_cl_{k}'] > compo.get(k,0)]['time'].sum() / _tot_time)
        return tree_leaves    
    
    def find_best_addition(self, 
                           compo:dict={}, 
                           significant_only:bool = False,
                           alpha:float=0.05,
                           freq_penalty: float=0, 
                           weight:bool = True, 
                           size:int = 1000, 
                           target:str = 'pm_per_48min', 
                           verbose:bool = True):
        _tree_leaves = self.explore_tree(compo=compo)
        _incr = np.zeros(len(_tree_leaves))
        for i, (pl, freq) in enumerate(_tree_leaves.items()):
            _sw, _swo = self.boostrap(new_addition=pl, base_compo= compo, target = target, weight= weight, size = size)
            bootstraper  = self.bootstrap_mean_delta(sample_w= _sw, sample_wo= _swo, n_bootstrap= size, weight_samples= weight)
            _incr[i] = bootstraper['mean_delta'] * (int(bootstraper['p-value'] <= alpha) if significant_only else 1) * freq ** freq_penalty
        new_added_player = list(_tree_leaves.keys())[np.argmax(_incr)]
        if verbose :
            print(f"Adding a player from cluster {new_added_player:>2} with estimated increase of {_incr.max():>5.2f}")  
        return new_added_player           
    
    def bootstrap_mean_delta(self, 
                            sample_w, 
                            sample_wo, 
                            n_bootstrap:int=1000, 
                            weight_samples:bool = True):
        n_w, n_wo = sample_w.shape[0], sample_wo.shape[0]
        if n_w ==0 or n_wo ==0:
            return {'mean_delta':0, 'p-value':1}
        else :
            observed_mean_delta = (sample_w['pm_per_48min'] * sample_w['time']).sum() / sample_w['time'].sum() \
                - (sample_wo['pm_per_48min'] * sample_wo['time']).sum() / sample_wo['time'].sum()
            _law_w = self._sample_distribution(sample_w, weight=weight_samples, target= 'pm_per_48min')
            _law_wo = self._sample_distribution(sample_wo, weight=weight_samples, target= 'pm_per_48min')
            bootstrap_stats = np.zeros(n_bootstrap)
            for i in range(n_bootstrap):
                boot_w = np.random.choice(a = _law_w.values[:,1], size= n_w, replace = True, p = _law_w.values[:,0])
                boot_wo = np.random.choice(a = _law_wo.values[:,1], size= n_wo, replace = True, p = _law_wo.values[:,0])
                bootstrap_stats[i] = boot_w.mean() - boot_wo.mean()
            shifted = bootstrap_stats - observed_mean_delta
            p_value = np.mean(np.abs(shifted) >= np.abs(observed_mean_delta))
            return {'mean_delta':observed_mean_delta, 'p-value':p_value}

    
    def empiric_means_delta(self, new_addition:int, base_compo:dict={}, target:str = 'pm_per_48min', weight:bool = True):
        _sw, _swo = self._create_samples_data(new_addition= new_addition, base_compo= base_compo)
        _lw = self._sample_distribution(sample_df= _sw, target= target, weight= weight)
        _lwo = self._sample_distribution(sample_df= _swo, target= target, weight= weight)
        return (_lw['weight'] * _lw[target]).sum() / _lw['weight'].sum() - (_lwo['weight'] * _lwo[target]).sum() / _lwo['weight'].sum()
    
    @staticmethod
    def _build_dict_from_tuple(tp:tuple):
        _dict = {}
        for item in tp:
            _dict[item] = _dict.get(item,0) +1
        return _dict 
    

    def find_compositions_stats(self, 
                                marginal_only:bool=False, 
                                bootstrap:bool = False, 
                                order:int=1, 
                                validity_threshold:int=30, 
                                alpha:float=0.05, 
                                weight_samples:bool = True, 
                                size:int = 1000):
        n_poss, n_valid, n_sign, n_valid_sign = 0, 0, 0, 0
        results = []
        if marginal_only :
            _candidates_compo = product(self.clusters, repeat= order)
        else :
            _candidates_compo = combinations_with_replacement(self.clusters, order)
        for compo in _candidates_compo:
            if marginal_only:
                base_compo = self._build_dict_from_tuple(compo[:-1])
                _df_w, _df_wo = self._create_samples_data(new_addition = compo[-1], base_compo= base_compo)
            else :
                _compo = self._build_dict_from_tuple(compo)
                _df_w = self._create_base_sample_df(compo = _compo)
                _df_wo = self.df.drop(index=_df_w.index)
            if bootstrap:
                bootstrapper = self.bootstrap_mean_delta(sample_w= _df_w, sample_wo= _df_wo, n_bootstrap=size, weight_samples= weight_samples)
                mean_diff = bootstrapper['mean_delta']                  
                pval = bootstrapper['p-value']
            else :
                t_test = ttest_ind(_df_w['pm_per_48min'], _df_wo['pm_per_48min'])
                pval = t_test.pvalue
                mean_diff = _df_w['pm_per_48min'].mean() - _df_wo['pm_per_48min'].mean()
            n_poss += 1
            n_valid += int(_df_w.shape[0]>= validity_threshold)
            n_sign += int (pval <= alpha and _df_w.shape[0] > 0)
            n_valid_sign += int(pval <= alpha and _df_w.shape[0] >= validity_threshold)
            if _df_w.shape[0] > 0:
                results.append({'id':'_'.join([f'{c}' for c in compo]),
                                        'significant' : pval <= alpha,
                                        'valid' : _df_w.shape[0]>= validity_threshold, 
                                        'mean_diff':mean_diff, 
                                        'pvalue':pval})
        stats = {'Level': order, 
                'n_poss': n_poss, 
                'n_valid': n_valid, 
                'n_sign': n_sign, 
                'n_valid_sign':n_valid_sign }
        return stats, results 
    