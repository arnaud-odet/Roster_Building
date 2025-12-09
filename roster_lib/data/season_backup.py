import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial
import itertools
import math
import copy

from roster_lib.constants import RAW_DATA_PATH, PREPROC_DATA_PATH
from roster_lib.utils.scraping import patient_souper
from roster_lib.scraping.starting_lineups import increment_lineups

DEFAULT_N_PROCESSORS = 24
MAX_RETRIEVAL_ATTEMPTS = 6
SUCCESS_PATH = PREPROC_DATA_PATH / 'team_success.csv'

TEAM_MAPPING = {'NOP': 'NOP',
 'LAL': 'LAL',
 'CHI': 'CHI',
 'DET': 'DET',
 'CLE': 'CLE',
 'MIN': 'MIN',
 'MEM': 'MEM',
 'BOS': 'BOS',
 'WAS': 'WAS',
 'NYK': 'NYK',
 'OKC': 'OKC',
 'SAC': 'SAC',
 'DEN': 'DEN',
 'ATL': 'ATL',
 'MIL': 'MIL',
 'LAC': 'LAC',
 'TOR': 'TOR',
 'DAL': 'DAL',
 'PHO': 'PHX',
 'POR': 'POR',
 'UTA': 'UTA',
 'MIA': 'MIA',
 'PHI': 'PHI',
 'ORL': 'ORL',
 'IND': 'IND',
 'GSW': 'GSW',
 'BRK': 'BKN',
 'CHO': 'CHA',
 'HOU': 'HOU',
 'SAS': 'SAS'}

TEAM_IDENTIFICATION_COL = ['Shooter', 'Assister', 'Fouler', 'Rebounder', 'ViolationPlayer', 
                       'FreeThrowShooter', 'LeaveGame', 'TurnoverPlayer']
OPPONENT_IDENTIFICATION_COL = ['Fouled', 'Blocker', 'TurnoverCauser']

TEAM_COL_FOULS = ['offensive','loose ball']
OPPONENT_COL_FOULS = ['personal','personal take','shooting', 
                    'away from play', 'clear path', 'flagrant','def 3 sec tech',
                    'personal block','offensive charge','shooting block','inbound']

MANUAL_CORRECTIONS = {
    # 2015
    '2015-10-30_DET_vs_CHI' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['K. Caldwell-Pope - caldwke01'] }], # Present all period without recording stat
    '2015-12-14_DET_vs_LAC' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['M. Morris - morrima03'] }], # Present all period without recording stat
    '2016-01-03_NYK_vs_ATL' : [{'quarter' : 2, 'team':'home', 'remove': ['A. Afflalo - afflaar01', 'J. Calderón - caldejo01', 'R. Lopez - lopezro01' ], 'add': [] }, # Changes not recorded
                               {'quarter' : 2, 'team':'away', 'remove': ['J. Teague - teaguje01', 'P. Millsap - millspa01', 'K. Bazemore - bazemke01', 'T. Sefolosha - sefolth01','E. Tavares - tavarwa01'], 'add': ['L. Patterson - pattela01'] }],# Changes not recorded
    '2016-01-04_MIA_vs_IND' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['G. Dragić - dragigo01'] }], # Present all period without recording stat
    '2016-01-05_DAL_vs_SAC' : [{'quarter' : 6, 'team':'home', 'remove': [], 'add': ['W. Matthews - matthwe02'] }], # Present all period without recording stat
    '2016-01-08_NOP_vs_IND' : [{'quarter' : 2, 'team':'home', 'remove': ['R. Stuckey - stuckro01'], 'add': [] }], # Foul by a player on his teammate
    '2016-01-14_PHI_vs_CHI' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['H. Thompson - thompho01'] }], # Present all period without recording stat
    '2016-01-25_SAC_vs_CHA' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['R. Gay - gayru01'] }], # Present all period without recording stat
    '2016-02-01_IND_vs_CLE' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['J. Smith - smithjr01'] }], # Present all period without recording stat
    '2016-02-09_DAL_vs_UTA' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['Z. Pachulia - pachuza01'] }], # Present all period without recording stat
    '2016-03-12_TOR_vs_MIA' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['J. Winslow - winslju01'] }], # Present all period without recording stat
    '2016-04-10_PHI_vs_MIL' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['T. Ennis - ennisty01'] }], # Present all period without recording stat
    '2016-04-24_BOS_vs_ATL' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['J. Crowder - crowdja01'] }], # Present all period without recording stat
    '2016-05-09_POR_vs_GSW' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['A. Crabbe - crabbal01'] }], # Present all period without recording stat
    # 2016
    '2016-11-28_WAS_vs_SAC' : [{'quarter' : 4, 'team':'away', 'remove': [], 'add': ['G. Temple - templga01'] }], # Present all period without recording stat
    '2016-11-30_OKC_vs_WAS' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['A. Roberson - roberan03'] }, # Present all period without recording stat
                               {'quarter' : 5, 'team':'away', 'remove': [], 'add': ['O. Porter - porteot01'] }], # Present all period without recording stat
    '2016-12-20_MIL_vs_CLE' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['J. Henson - hensojo01'] }], # Present all period without recording stat
    '2017-01-08_POR_vs_DET' : [{'quarter' : 6, 'team':'home', 'remove': [], 'add': ['A. Crabbe - crabbal01'] }], # Present all period without recording stat
    '2017-03-10_SAC_vs_WAS' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['W. Cauley-Stein - caulewi01'] }], # Present all period without recording stat
    '2017-03-11_CHA_vs_NOP' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['S. Hill - hillso01'] }], # Present all period without recording stat
    '2017-05-02_BOS_vs_WAS' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['O. Porter - porteot01'] }], # Present all period without recording stat
    # 2017
    '2017-10-30_MEM_vs_CHA' : [{'quarter' : 3, 'team':'home', 'remove': [' - NULL'], 'add': [] }, # Removing NULL 
                               {'quarter' : 3, 'team':'away', 'remove': [' - NULL'], 'add': [] }], # Removing NULL 
    '2017-11-20_DAL_vs_BOS' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['A. Horford - horfoal01'] }], # Present all period without recording stat
    '2017-12-09_MEM_vs_OKC' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['S. Adams - adamsst01'] }], # Present all period without recording stat
    '2018-01-06_MIN_vs_NOP' : [{'quarter' : 2, 'team':'away', 'remove': [], 'add': ['D. Cunningham - cunnida01'] }], # Present all period without recording stat
    '2018-01-22_NOP_vs_CHI' : [{'quarter' : 6, 'team':'home', 'remove': [], 'add': ['D. Miller - milleda01'] }], # Present all period without recording stat
    '2018-03-07_DET_vs_TOR' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['K. Lowry - lowryky01'] }], # Present all period without recording stat
    '2018-03-26_CHA_vs_NYK' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['C. Lee - leeco01'] }], # Present all period without recording stat
    '2018-03-30_LAL_vs_MIL' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['J. Terry - terryja01'] }], # Present all period without recording stat
    '2018-04-25_OKC_vs_UTA' : [{'quarter' : 4, 'team':'away', 'remove': [], 'add': ['J. Ingles - inglejo01'] }], # Present all period without recording stat
    # 2018
    '2018-11-05_NYK_vs_CHI' : [{'quarter' : 6, 'team':'home', 'remove': [], 'add': ['N. Vonleh - vonleno01'] }, # Present all period without recording stat
                               {'quarter' : 6, 'team':'away', 'remove': [], 'add': ['J. Holiday - holidju01'] }], # Present all period without recording stat
    '2018-11-16_BOS_vs_TOR' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['M. Morris - morrima03'] }], # Present all period without recording stat
    '2018-11-23_DEN_vs_ORL' : [{'quarter' : 3, 'team':'home', 'remove': [], 'add': ['J. Hernangómez - hernaju01'] }], # Present all period without recording stat
    '2018-12-07_BKN_vs_TOR' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['J. Harris - harrijo01'] }], # Present all period without recording stat
    '2018-12-22_WAS_vs_PHX' : [{'quarter' : 6, 'team':'home', 'remove': [], 'add': ['J. Green - greenje02'] }], # Present all period without recording stat
    '2019-01-03_GSW_vs_HOU' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['P. Tucker - tuckepj01'] }], # Present all period without recording stat
    '2019-01-04_CHI_vs_IND' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['W. Carter - cartewe01'] }], # Present all period without recording stat
    '2019-01-10_SAS_vs_OKC' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['T. Ferguson - fergute01'] }], # Present all period without recording stat
    '2019-02-22_OKC_vs_UTA' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['J. Ingles - inglejo01'] }], # Present all period without recording stat
    '2019-03-26_CHA_vs_SAS' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['M. Bridges - bridgmi02'] }], # Present all period without recording stat
    '2019-03-29_MIN_vs_GSW' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['A. Iguodala - iguodan01'] }], # Present all period without recording stat
    # 2019
    '2019-10-25_DEN_vs_PHX' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['M. Beasley - beaslma01'] }], # Present all period without recording stat
    '2019-11-08_MIN_vs_GSW' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['T. Graham - grahatr01'] }], # Present all period without recording stat
    '2019-11-29_IND_vs_ATL' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['D. Hunter - huntede01'] }], # Present all period without recording stat
    '2019-12-18_WAS_vs_CHI' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['I. Smith - smithis01'] }], # Present all period without recording stat
    '2019-12-31_SAS_vs_GSW' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['D. Lee - leeda03'] }], # Present all period without recording stat
    '2020-01-09_DET_vs_CLE' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['T. Snell - snellto01'] }], # Present all period without recording stat
    '2020-01-27_MIN_vs_SAC' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['H. Barnes - barneha02'] }], # Present all period without recording stat
    '2020-02-09_ATL_vs_NYK' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['D. Hunter - huntede01'] }], # Present all period without recording stat
    '2020-08-08_DEN_vs_UTA' : [{'quarter' : 6, 'team':'home', 'remove': [], 'add': ['M. Morris - morrimo01'] }], # Present all period without recording stat
    '2020-09-09_BOS_vs_TOR' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['O. Anunoby - anunoog01'] }], # Present all period without recording stat
    # 2020
    '2020-12-26_DET_vs_CLE' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['D. Wright - wrighde01'] }], # Present all period without recording stat
    '2021-01-04_NOP_vs_IND' : [{'quarter' : 5, 'team':'away', 'remove': [], 'add': ['J. Holiday - holidju01'] }], # Present all period without recording stat
    '2021-01-07_DEN_vs_DAL' : [{'quarter' : 5, 'team':'home', 'remove': [], 'add': ['G. Harris - harriga01'] }], # Present all period without recording stat
}

NON_EXISTING_OVERTIMES = {
    # 2016
    '2016-10-30_HOU_vs_DAL' : 5,
    '2016-12-02_SAS_vs_WAS' : 5,
    '2017-01-27_IND_vs_SAC' : 6,
}

IDENTIFIED_BAD_CHANGES_INDICES = {
    2015 : [],
    2016 : [261765], # Non existing
    2017 : [9267, # Non existing
            23311, # Non existing
            23312, # Non existing
            23324, # Duplicated
            23333, # Duplicated
            23336, # Duplicated
            26437, # Duplicated
            26438, # Duplicated
            26439, # Duplicated
            36223, # Non existing
            36225, # Non existing
            359438, # Duplicated
            462022 # Non existing
            ],
    2018 : [],
    2019 : [],
    2020 : []
}

class Season :
    
    def __init__(self, year:int, delay:int = 0, verbose_level:int=0, keep_playoffs:bool=True):
        self.year = year
        season_str = f"{year}-{str(year-year//100*100+1).zfill(2)[-2:]}"
        self.delay = delay
        self.verbose_level = verbose_level
        self.raw_pbp_path = RAW_DATA_PATH / 'play_by_play' / f'NBA_PBP_{season_str}.csv'
        self.starting_lineups_path = PREPROC_DATA_PATH / 'starting_lineups' / f'{season_str}.csv'
        self.identifications_path = PREPROC_DATA_PATH / 'identifications' / f'identifications_{season_str}.pkl'
        self.stats_path = PREPROC_DATA_PATH / 'stats' / f'{season_str}.csv'
        self.bad_changes_path = PREPROC_DATA_PATH / 'stats' / f'bad_changes_{season_str}.pkl'
        
        self._build_architecture()
        
        print("Loading play-by-play data ...", end = '\r')
        self.pbp = self._load_pbp_data()
        self.date_list = self.pbp['date'].drop_duplicates().tolist()
        print("Loading play-by-play data COMPLETED", end = '\n')
        
        print("Processing play-by-play data ...", end = '\r')            
        self._process_pbp_data()        
        print("Processing play-by-play data COMPLETED", end = '\n')    
                      
        self.starting_lineups = self._load_starting_lineups()
        if not keep_playoffs :
            self.pbp = self.pbp[self.pbp['GameType']=='regular']
        self._identify_on_court_players()       

        self.stats, self.bad_changes = self._load_stats()  
        print("Processing lineups statistics ...", end = '\r')             
        self.team_lineup_stats = self._build_team_lineup_stats_df() 
        self._attribute_lineups_id()
        if not os.path.exists(self.stats_path):
            self.stats.to_csv(self.stats_path)              
        self.rosters = self._build_rosters()
        print("Processing lineups statistics COMPLETED", end = '\n')             

        self.success = self._load_sucess()
    
    # Main functions
    def _build_architecture(self):
        for filepath in [self.starting_lineups_path, self.identifications_path, self.stats_path]:
            if not os.path.exists(filepath.parent) :
                os.makedirs(filepath.parent)
        
    def _load_pbp_data(self):

        return pd.read_csv(self.raw_pbp_path)

    def _load_starting_lineups(self): # DEPRECATED
        i = 0
        print("Attempting to load starting lineups ...", end = '\r')
        while not self._check_retrieval_completion() and i < MAX_RETRIEVAL_ATTEMPTS :
            print("Attempting to load starting lineups FAILED - Retrieving starting lineups instead ...")
            i+=1
            print(f"Starting lineups retrieval attempt n° {i:>2} in progress ..." + " "*50)
            self._retrieve_starting_lineups(n_processors=1)       
        if i == 0 : 
            print("Attempting to load starting lineups SUCCEEDED")   
        elif self._check_retrieval_completion():
            print("Starting lineups retrieval SUCCEEDED")
        else :
            print("Starting lineups retrieval FAILED - MAX_RETRIEVAL_ATTEMPS reached, season.starting_lineups is likely incomplete")
        return pd.read_csv(self.starting_lineups_path, index_col='game_id') 
        
    def _process_pbp_data(self):
        # Basic preprocessing
        self.pbp['date'] = pd.to_datetime(self.pbp['Date'], format='%B %d %Y')
        self.pbp['HomeScore'] = self.pbp['HomeScore'].astype(int)
        self.pbp['AwayScore'] = self.pbp['AwayScore'].astype(int)
        self.pbp = self.pbp.drop(columns=['Date','URL','Location','Time','WinningTeam'])
        
        self.pbp['AwayTeam'] = self.pbp['AwayTeam'].map(TEAM_MAPPING)
        self.pbp['HomeTeam'] = self.pbp['HomeTeam'].map(TEAM_MAPPING)
        self.pbp['game_id'] = self.pbp.apply(lambda row : f"{row['date'].year}-{str(row['date'].month).zfill(2)}-{str(row['date'].day).zfill(2)}_{row['HomeTeam']}_vs_{row['AwayTeam']}", axis = 1)
        self.pbp['time_left'] = self.pbp.apply(lambda row : self._compute_time_left(row['Quarter'], row['SecLeft']), axis = 1)        
        self.pbp['team'] = ['away' if binary else 'home' for binary in self.pbp['HomePlay'].isna()]
        self.pbp['diff_score'] = self.pbp['HomeScore'] - self.pbp['AwayScore']
        self.pbp_processed = True
        
    def _identify_on_court_players(self):        
        # Changes handling
        self.pbp['change'] = self.pbp['LeaveGame'].isna() == False
        self.quarter_df = self._create_quarter_df(self.pbp)
        self.pbp = self.pbp.merge(self.quarter_df, left_on= ['game_id','Quarter'], right_index=True)
        self.pbp['before_first_change'] = self.pbp['time_left'] >= self.pbp['first_change']
        if os.path.exists(self.identifications_path):
            with open(self.identifications_path, 'rb') as f:
                self.identifications = pickle.load(f)
        else : 
            self.identifications = self._process_identifications()        

    def _load_stats(self):
        print("Attempting to load lineups statistics ...", end = '\r')        
        if os.path.exists(self.stats_path) and os.path.exists(self.bad_changes_path):
            stats_df = pd.read_csv(self.stats_path, index_col=0)
            with open(self.bad_changes_path, 'rb') as f :
                errors = pickle.load(f)
            print("Attempting to load lineups statistics SUCCEEDED", end = '\n')   
        else :
            print("Attempting to load lineups statistics FAILED - Building statistics instead", end = '\n')
            stats_df , errors = self._process_stats()
            stats_df.reset_index(drop=True, inplace=True)
            with open(self.bad_changes_path, 'wb') as f :
                pickle.dump(errors, f)
            
        return stats_df, errors

    def _build_team_lineup_stats_df(self):
        team_lineup_stats = {}
        for team in self.stats['team_id'].unique():       
            df = self.stats[self.stats['team_id']==team][['lineup','time','pm']].groupby('lineup').sum().sort_values('time',ascending = False)
            pl_list = list(set(sum([ind.split('_')for ind in df.index],[])))
            for pl in pl_list :
                df[pl] = 0
            for ind in df.index :
                for pl in ind.split('_'):
                    df.loc[ind,pl] = df.loc[ind,'time']
            ordered_columns = list(df.sum(axis = 0)[pl_list].sort_values(ascending=False).index)
            df[ordered_columns] = (df[ordered_columns] > 0).astype(int) 
            df['pm_per_48min'] = df['pm'] / (df['time'] / (48*60))
            df.reset_index(inplace=True)
            df['id'] = [f"{team}_{self.year}_{str(ind+1).zfill(3)}" for ind in df.index]
            team_lineup_stats[team] = df[['id','lineup','time','pm', 'pm_per_48min'] + ordered_columns]
            
        return team_lineup_stats
    
    def _attribute_lineups_id(self):
        if not 'id' in self.stats.columns :
            lu_id_df = pd.concat([df[['lineup','id']] for df in self.team_lineup_stats.values()])
            self.stats = pd.merge(self.stats, lu_id_df, left_on = 'lineup', right_on = 'lineup')
            
    def _build_rosters(self):
        rosters = {}
        for t, df in self.team_lineup_stats.items():
            _tdf = df.copy()
            pl_cols = _tdf.drop(columns = ['id','lineup','time','pm','pm_per_48min']).columns
            for pl_col in pl_cols :
                _tdf[pl_col] = _tdf['time'] * _tdf[pl_col]
            roster = dict(_tdf[pl_cols].sum(axis = 0))
            rosters[t] = roster
        return rosters

    def _load_sucess(self):
        sdf = pd.read_csv(SUCCESS_PATH)
        sdf = sdf[sdf['YEAR'] == self.year].set_index('TEAM')
        return sdf

    # Auxiliaries
  
    def _check_retrieval_completion(self): # DEPRECATED
        try :
            df = pd.read_csv(self.starting_lineups_path, index_col = 'game_id')
            if 'home_starter_1' in df.columns:
                n_dates = df['date'].nunique()
                n_games = df.shape[0]
                n_games_retrieved = df[df['home_starter_1'].isna() == False].shape[0]
                if n_dates == len(self.date_list) and n_games == n_games_retrieved :
                    return True
                else :
                    return False
            else :
                return False
        except :
            # Case the starting lineup filepath does not exist
            return False         
        
    def _retrieve_starting_lineups(self, n_processors=None): # DEPRECATED

        if n_processors is None:
            n_processors = max(1, cpu_count() - 1)  # Leave one CPU free
        print(f"Processing {len(self.date_list)} dates using {n_processors} processors...")
        
        if n_processors == 1 :
            # Sequential execution to test code
            stat_str = ""
            for i, date in enumerate(self.date_list) :
                date_str = f"{date.year}-{date.month:>2}-{date.day:>2}"
                print(f"Processing date n° {i+1:>2} of {len(self.date_list)} : {date_str} {stat_str}", end = '\r')
                increment_lineups(filepath= self.starting_lineups_path, date = date, delay=self.delay, verbose_level=self.verbose_level)
                ludf = pd.read_csv(self.starting_lineups_path)
                stat_str = f"- {ludf.shape[0]:>4} games found"
                try :
                    stat_str += f" of which {ludf[ludf['home_starter_1'].isna() == False].shape[0]:>4} games retrieved."
                except :
                    pass             
            
        else :
            args = [(self.starting_lineups_path, date, self.delay, self.verbose_level) for date in self.date_list]
            # Create pool and process with progress bar
            with Pool(processes=n_processors) as pool:
                list(tqdm(
                    pool.imap(self._process_date, args),
                    total=len(args),
                    desc="Processing dates ..."
                ))

            print("Processing dates over !")

    def _process_identifications(self, n_processors = None):
        
        if n_processors == None :
            n_processors = DEFAULT_N_PROCESSORS
        print("Preparing data for players identifications ...", end = '\r')             
        args = [(self.pbp[self.pbp['game_id']==g_id].copy(), g_id) for g_id in self.pbp['game_id'].unique()]
        print("Preparing data for players identifications COMPLETED", end = '\n')             
        
        # Process in parallel
        with Pool(processes=n_processors) as pool:
            results = list(tqdm(
                pool.imap(self._process_game_identifications, args),
                total=len(args),
                desc="Identifying players on court"
            ))
        identifications = dict(results)
        
        # Process manual corrections 
        for g, corr_list in MANUAL_CORRECTIONS.items():
            if g in identifications.keys():
                for corr in corr_list :
                    q = corr['quarter']
                    t = corr['team']
                    init_list = identifications[g][q][t]['starting']
                    identifications[g][q][t]['starting'] = [pl for pl in init_list + corr['add'] if pl not in corr['remove']]
        for g,q in NON_EXISTING_OVERTIMES.items():
            if g in identifications.keys():
                identifications[g].pop(q)
                    
        with open(self.identifications_path, 'wb') as f:
            pickle.dump(identifications, f, protocol= pickle.HIGHEST_PROTOCOL)
        return identifications

    def _process_stats(self, n_processors = None):
        
        if n_processors == None :
            n_processors = DEFAULT_N_PROCESSORS
        print("Preparing data for game statistics ...", end = '\r')             
        args = [(self.pbp[self.pbp['game_id']==g_id].copy(), 
                 self.identifications[g_id] ,
                 g_id,
                 IDENTIFIED_BAD_CHANGES_INDICES[self.year]) for g_id in self.pbp['game_id'].unique()]
        print("Preparing data for game statistics COMPLETED", end = '\n')             
        
        # Process in parallel
        with Pool(processes=n_processors) as pool:
            results = list(tqdm(
                pool.imap(self._process_game_stats, args),
                total=len(args),
                desc="Buidling game statistics"
            ))
        
        stats = pd.concat([item[0] for item in results])
        errors = [item[1] for item in results if item != []]
        errors = list(itertools.chain.from_iterable(errors))
        
        return stats, errors
        
    # Parrallel processing
    @staticmethod
    def _process_date(args):
        fp, date, delay, verbose_level = args
        increment_lineups(filepath= fp, date = date, delay = delay, verbose_level = verbose_level)
    
    @staticmethod
    def _process_game_identifications(args):
        gdf, g_id = args  
        identification = {}
        # Removing foull type that do not indicate presence on the floor (technical fouls) and misrecorded (clear path)
        gdf = gdf[gdf['FoulType'].isin(['clear path','technical']) ==False].copy()
        gdf['Quarter'] = gdf['Quarter'].astype(int)
        for q in gdf['Quarter'].unique():
            q_df = gdf[gdf['Quarter'] == q].copy()
            q_home, q_away = Season._create_player_list(q_df)
            s5h, s5a = Season._find_starting_lineup(q_home, q_df, q), Season._find_starting_lineup(q_away, q_df, q)
            identification[q] = {'home' : {'starting' : s5h, 'all' : q_home}, 'away': {'starting' : s5a, 'all' : q_away}}  
            
        return (g_id, identification)

    @staticmethod
    def _process_game_stats(args):
        g_df, g_ident ,gid, identified_bad_indices = args        
        stats = []
        errors = []
        home_team_id, away_team_id = gid[-10:-7], gid[-3:] 
        for q in g_ident.keys():
            gq_df = g_df[(g_df['Quarter']==q) & (g_df['change'])].copy()
            lineups = {team : g_ident[q][team]['starting'] for team in ['home','away']}
            for lineup in lineups.values():
                lineup.sort()
            for team in ['home','away']: 
                start_time = 720 if q <= 4 else 300
                start_diff, end_diff = g_df[g_df['Quarter'] == q].iloc[0]['diff_score'], g_df[g_df['Quarter'] == q].iloc[-1]['diff_score']
                lineup = copy.deepcopy(lineups[team])
                loc_mult = -1 if team == 'away' else 1
                team_id = away_team_id if team == 'away' else home_team_id
                for index, row in gq_df[gq_df['team']==team][['SecLeft']].drop_duplicates().iterrows():
                    time = row['SecLeft']
                    if time != start_time and time != 0 : 
                        diff = gq_df.loc[index,'diff_score']
                        # Step 1 : record +/-
                        stats.append({'game_id': gid, 'quarter': q, 'team_id': team_id,'team': team,'lineup': '_'.join(copy.deepcopy(lineup)), 'time': start_time - time, 'pm': loc_mult * (diff - start_diff)})
                        # Step 2 : update references
                        start_time = time
                        start_diff = diff
                        for pl_index, pl_row in gq_df[(gq_df['team']==team) & (gq_df['SecLeft']==time)][['EnterGame','LeaveGame']].iterrows():
                            if pl_row['LeaveGame'] in lineup and not pl_index in identified_bad_indices :
                                # Record the change if the player going out is recorded on the field earlier
                                lineup.append(pl_row['EnterGame'])
                                lineup.remove(pl_row['LeaveGame'])
                            elif pl_index not in identified_bad_indices :
                                # Record the change as erroneous change
                                errors.append({'game_id': gid, 'line_ind': pl_index,'quarter' : q, 'time' : time, 'out':pl_row['LeaveGame'], 'in': pl_row['EnterGame'] })
                            
                        lineup.sort()
                # Step 3 : appending the last lineup
                stats.append({'game_id': gid, 'quarter': q, 'team_id': team_id,'team': team,'lineup': '_'.join(copy.deepcopy(lineup)), 'time': start_time, 'pm': loc_mult * (end_diff - start_diff)})

        return pd.DataFrame(stats), errors


    # Methods
    @staticmethod
    def _compute_time_left(quarter, seconds_left):
        return (4-quarter) * (12 if quarter <= 4 else 5) * 60 + seconds_left

    @staticmethod
    def _create_quarter_df(df):
        quarter_df = df[df['change']][["game_id",'Quarter','time_left']]\
            .groupby(["game_id",'Quarter']).max()\
            .rename(columns ={'time_left':'first_change'})        
        return quarter_df

    @staticmethod
    def _create_player_list(temporal_df):
        df = temporal_df.copy()

        # Identification from team actions 
        home_list = list(df[(df['team'] == 'home') & (df['FoulType'].isin(OPPONENT_COL_FOULS) == False)][TEAM_IDENTIFICATION_COL].values.flatten())
        away_list = list(df[(df['team'] == 'away') & (df['FoulType'].isin(OPPONENT_COL_FOULS) == False)][TEAM_IDENTIFICATION_COL].values.flatten())
        # Identification from opponent actions
        home_list += list(df[(df['team'] == 'away') & (df['FoulType'].isin(OPPONENT_COL_FOULS) == False)][OPPONENT_IDENTIFICATION_COL].values.flatten())        
        away_list += list(df[(df['team'] == 'home') & (df['FoulType'].isin(OPPONENT_COL_FOULS) == False)][OPPONENT_IDENTIFICATION_COL].values.flatten())
        # Identification for fouls recorded as opponent actions
        home_list += df[(df['team'] == 'away') & (df['FoulType'].isin(TEAM_COL_FOULS) == False)]['Fouler'].values.tolist()                
        away_list += df[(df['team'] == 'home') & (df['FoulType'].isin(TEAM_COL_FOULS) == False)]['Fouler'].values.tolist()
        # Identification of fouled players from team actions
        home_list += df[(df['team'] == 'home') & (df['FoulType'].isin(OPPONENT_COL_FOULS))]['Fouled'].values.tolist()                
        away_list += df[(df['team'] == 'away') & (df['FoulType'].isin(OPPONENT_COL_FOULS))]['Fouled'].values.tolist()        
        # Removing duplicates and 'Team' value
        home_list = list(set(home_list))
        away_list = list(set(away_list))
        home_list = [pl for pl in home_list if (type(pl) == str and pl != 'Team')]
        away_list = [pl for pl in away_list if (type(pl) == str and pl != 'Team')]

        # In case of missing players, look into jumpball - jumpball identification contains errors in the pbp
        if len(home_list) <5 :
            if df[df['JumpballHomePlayer'].isna() == False].shape[0] > 0 :
                home_jumper = df[df['JumpballHomePlayer'].isna() == False]['JumpballHomePlayer'].values.tolist()[0] 
                if home_jumper not in home_list:
                    home_list.append(home_jumper)
        if len(away_list) <5 :
            if df[df['JumpballAwayPlayer'].isna() == False].shape[0] > 0 :
                away_jumper = df[df['JumpballAwayPlayer'].isna() == False]['JumpballAwayPlayer'].values.tolist()[0] 
                if away_jumper not in away_list:
                    away_list.append(away_jumper)

        return home_list, away_list    

    @staticmethod
    def _find_starting_lineup(pl_list, q_df, quarter):
        max_time = 720 if quarter <= 4 else 300
        s5_list = []
        for pl in pl_list : 
            first_id = q_df[((q_df[TEAM_IDENTIFICATION_COL + OPPONENT_IDENTIFICATION_COL] == pl).any(axis = 1)) & (q_df['SecLeft'] < max_time)]['SecLeft'].max()                
            incoming = q_df[(q_df['EnterGame'] == pl) & (q_df['SecLeft'] < max_time)]['SecLeft'].max()
            # Correction for player recorded as entering and leaving game at the same first time
            outgoing = q_df[(q_df['LeaveGame'] == pl) & (q_df['SecLeft'] < max_time)]['SecLeft'].max()
            if incoming == outgoing and not math.isnan(incoming) and not math.isnan(outgoing) :
                in_index = q_df[(q_df['EnterGame'] == pl) & (q_df['SecLeft'] < max_time)].index[0]
                out_index = q_df[(q_df['LeaveGame'] == pl) & (q_df['SecLeft'] < max_time)].index[0]
                first_change_correction = out_index < in_index
            else :
                first_change_correction = False
            s5_pl = (math.isnan(incoming) or (first_id > incoming) or first_change_correction) and not math.isnan(first_id) 
            # if quarter == 5 :
            #     print(f"{quarter=} | {pl=} | {first_id=} | {incoming=} | {s5_pl=}")
            if s5_pl :
                s5_list.append(pl)
        return s5_list

    # Visible
    def show_games(self, team):
        sldf = self.starting_lineups.copy()
        h_sldf = sldf[sldf['home'] == team].rename(columns = {f"home_starter_{i}" : f"starter_{i}" for i in range(1,6)})[[f"starter_{i}" for i in range(1,6)]]
        a_sldf = sldf[sldf['away'] == team].rename(columns = {f"away_starter_{i}" : f"starter_{i}" for i in range(1,6)})[[f"starter_{i}" for i in range(1,6)]]

        return sldf[['date','home','away','url']].merge(pd.concat([h_sldf,a_sldf]),
            left_index = True,
            right_index = True,
            how = 'right').sort_values('date')

    def investigate_quarter_starting_lineup(self, show_df = True):
        n_id = []
        for g_id, g_dict in self.identifications.items():
            for q, gq_dict in g_dict.items():
                n_id += [ { 'game_id': g_id,'quarter': q, 'team': 'home', 'n_id': len(gq_dict['home']['starting'])} , 
                        { 'game_id': g_id,'quarter': q, 'team': 'away', 'n_id': len(gq_dict['away']['starting'])} ]
        nid_df = pd.DataFrame(n_id)
        if show_df :
            display(nid_df.pivot_table(index = 'quarter', columns='n_id', values = 'team', aggfunc='count').fillna(0).astype(int))     
        return nid_df
 
    def show_incomplete_quarter_starting_lineups(self, show_df:bool = True):
        nid_df = self.investigate_quarter_starting_lineup(show_df = show_df)
        for index,row in nid_df[nid_df['n_id']!=5].sort_values('game_id').iterrows():
            g = row['game_id']
            q = row['quarter']
            t = row['team']
            url = self.starting_lineups.loc[g,'url'] + (f'?period=OT{q-4}' if q >4 else f'?period=Q{q}')
            l = self.identifications[g][q][t]['starting']
            all_q = [self.identifications[g][qt][t]['all'] for qt in range(1,max(q,4)+1)]
            print(f"Game: '{g}'", f"Quarter: {q}", url, t)
            print(t, len(l), f"players identified as starters {l}")
            print('Other identified players:',[p for p in list(set(itertools.chain.from_iterable(all_q))) if p not in l])
    
if __name__ == "__main__" :
    
    for yr in [2019,2018,2017,2016,2015,2020] :
        print(f"-----> PROCESSING SEASON {yr}-{yr-1999} <-----" + 60*" ")
        s = Season(year= yr, delay = 4, verbose_level=1)