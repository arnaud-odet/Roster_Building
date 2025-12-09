import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from pathlib import Path

from roster_lib.utils.scraping import patient_souper

PREPROC_DATA_PATH = Path('./preproc_data')
BASE_URI = "https://www.nba.com/standings"

team_wr_dict = {'Miami Heat': 'MIA',
 'Boston Celtics': 'BOS',
 'Milwaukee Bucks': 'MIL',
 'Philadelphia ers': 'PHI',
 'Toronto Raptors': 'TOR',
 'Chicago Bulls': 'CHI',
 'Brooklyn Nets': 'BKN',
 'Cleveland Cavaliers': 'CLE',
 'Atlanta Hawks': 'ATL',
 'Charlotte Hornets': 'CHA',
 'New York Knicks': 'NYK',
 'Washington Wizards': 'WAS',
 'Indiana Pacers': 'IND',
 'Detroit Pistons': 'DET',
 'Orlando Magic': 'ORL',
 'Phoenix Suns': 'PHX',
 'Memphis Grizzlies': 'MEM',
 'Golden State Warriors': 'GSW',
 'Dallas Mavericks': 'DAL',
 'Utah Jazz': 'UTA',
 'Denver Nuggets': 'DEN',
 'Minnesota Timberwolves': 'MIN',
 'LA Clippers': 'LAC',
 'New Orleans Pelicans': 'NOP',
 'San Antonio Spurs': 'SAS',
 'Los Angeles Lakers': 'LAL',
 'Sacramento Kings': 'SAC',
 'Portland Trail Blazers': 'POR',
 'Oklahoma City Thunder': 'OKC',
 'Houston Rockets': 'HOU',
 'Los Angeles Clippers': 'LAC',
 'Charlotte Bobcats': 'CHA',
 'New Orleans Hornets': 'NOP',
 'New Jersey Nets': 'BKN',
 'Seattle SuperSonics': 'OKC',
 'New Orleans/Oklahoma City Hornets': 'NOP',
 'Vancouver Grizzlies': 'MEM',
 'CHA':'CHA',
 'CHH':'NOP'}
 

def scrap_teams_wr_year(year:int):
    str_yr = f"?Season={year-1}-{str(year-2000).zfill(2)}"
    url = BASE_URI + str_yr
    soup = patient_souper(url, "StatsStandingsTable_row__o6A7G")
    wr_dict = {}
    for item in soup.find_all(class_ = "StatsStandingsTable_row__o6A7G"):
        tm_str = item.find(class_ = "Anchor_anchor__cSc3P StatsStandingsTable_teamLink__8f7tE").text
        tm_str = ''.join([i for i in tm_str if not i.isdigit()])
        team = tm_str.replace("\xa0",' ').split(' - ')[0]
        win_rate = float(item.find_all('td')[3].text)
        wr_dict[team] = win_rate
    wr_df = pd.DataFrame([wr_dict]).T.reset_index().rename(columns={0:'win_rate', 'index' : 'TEAM'})
    wr_df['YEAR'] = year
    return wr_df
    
def scrap_teams_wr(years:list):
    df_list = []
    for yr in years :
        try : 
            df_list.append(scrap_teams_wr_year(year=yr))
            print(f"Season {yr-1}/{str(yr-2000).zfill(2)} completed")
        except :
            print(f"--- Error in season {yr-1}/{str(yr-2000).zfill(2)}")
    return pd.concat(df_list).reset_index(drop=True)

def process_win_rate_df(win_rate_df):
    df = win_rate_df.copy()
    # Step 1 : handling the Charlotte Hornets case
    for ind, row in df.iterrows():
        if row['TEAM'] == "Charlotte Hornets" and row['YEAR'] < 2004:
            df.loc[ind,'TEAM'] = "New Orleans Hornets"
    # Step 2 : Mapping every other team
    df['TEAM'] = df['TEAM'].map(team_wr_dict)
    return df

def create_win_rate_df(years:list, save : bool = False):
    wrdf = scrap_teams_wr(years)
    p_wrdf = process_win_rate_df(wrdf)
    if save :
        p_wrdf.to_csv(PREPROC_DATA_PATH/"team_win_rates.csv")
    return p_wrdf
    