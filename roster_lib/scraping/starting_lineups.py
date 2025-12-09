import pandas as pd
import numpy as np
import datetime
import os

from roster_lib.utils.scraping import patient_souper

def find_starting_lineups(bx_url, delay:int=0):
    soup = patient_souper(bx_url, expected_class="GameBoxscoreTablePlayer_gbp__mPF20", base_delay=delay)
    starter_dict = {}
    i=0
    for item in soup.find_all(class_ = "GameBoxscoreTablePlayer_gbp__mPF20"):
        if item.find(class_="GameBoxscoreTablePlayer_gbpPos__KW2Nf").text != '' :
            loc = "away" if i // 5 == 0 else "home"
            dict_key = f"{loc}_starter_{i%5+1}"
            starter_dict[dict_key] = item.find(class_ = "GameBoxscoreTablePlayer_gbpNameShort__hjcGB").text
            i += 1
    return starter_dict    

def scrap_games(date:datetime.date, delay:int = 0):
    year, month, day = date.year, date.month, date.day
    games = []
    str_date = f"{year}-" + f"{month}".zfill(2) + "-" + f"{day}".zfill(2)
    url = f"https://www.nba.com/games?date={str_date}"
    soup = patient_souper(url, expected_class= 'GameCard_gcMain__q1lUW', base_delay=delay)
    for item in soup.find_all(class_ = 'GameCard_gcMain__q1lUW'):
        g_url = item.find_all(href=True)[0]['href']
        home = g_url[13:16].upper()
        away = g_url[6:9].upper()
        bx_url =  f"https://www.nba.com{g_url}/box-score"
        games.append({'game_id' : str_date + '_' + home + '_vs_' + away,'date':str_date, 'home':home, 'away':away, 'url': bx_url})
        
    return games


def increment_games(filepath, date:datetime.date, delay:int=0, verbose_level:int = 2):
    
    year, month, day = date.year, date.month, date.day
    str_date = f"{year}-" + f"{month}".zfill(2) + "-" + f"{day}".zfill(2)
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col='game_id')
        if str_date in df['date'].tolist() and verbose_level >= 2:
            print(f"Date {str_date} already registered" + " "*50)
        else : 
            try :
                df = pd.concat([df, pd.DataFrame(scrap_games(date=date, delay=delay)).set_index('game_id')])
                df.to_csv(filepath)

            except :
                if verbose_level >= 1:
                    print(f"Error while processing date {str_date}" + " "*50)
    else :
        try :
            df = pd.DataFrame(scrap_games(date=date, delay=delay)).set_index('game_id')
            df.to_csv(filepath)
        except :
            if verbose_level >= 1:
                print(f"Error while processing date {str_date}"+ " "*50 )        

    
        
def increment_lineups(filepath, date:datetime.date, delay:int=0, verbose_level:int = 2):
    
    year, month, day = date.year, date.month, date.day
    str_date = f"{year}-" + f"{month}".zfill(2) + "-" + f"{day}".zfill(2)
    
    # Verify presence of the game URLs
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col='game_id')
        if not str_date in df['date'].tolist() :
            increment_games(filepath=filepath, date=date, delay=delay, verbose_level = verbose_level)
    else :
        increment_games(filepath=filepath, date=date, delay=delay, verbose_level= verbose_level)
    
    df = pd.read_csv(filepath, index_col='game_id')
    # Iterate through game URLs
    for g_id, row in df[df['date'] == str_date].iterrows():
        bx_url = row['url']
        try :
            if not 'home_starter_1' in df.columns :
                lineup = pd.DataFrame([{'game_id':g_id} | find_starting_lineups(bx_url, delay=delay)]).set_index('game_id')
                df = df.merge(lineup, left_on = 'game_id', right_index = True, how = 'left')
                df.to_csv(filepath)
            else :
                if df.loc[g_id].isna().any() :
                    lineup = pd.DataFrame([{'game_id':g_id} | find_starting_lineups(bx_url, delay=delay)]).set_index('game_id')
                    df.update(lineup)
                    df.to_csv(filepath)                
                elif verbose_level >=2:
                    print(f"Game {g_id} already retrieved" + " "*50)
        except :
            if verbose_level >=1:
                print(f"Error while processing game {g_id}" + " "*50)
        
    df.to_csv(filepath)