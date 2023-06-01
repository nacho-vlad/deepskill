import pandas as pd
from datetime import datetime
from collections import defaultdict

def clean_dataframe(df):    
    df = df[df['Result'].isin(["1-0", "0-1", "1/2-1/2"])]
    df = df[df.WhiteElo != '?']
    df = df[df.BlackElo != '?']
    return df

def sort_by_time(df):
    df["time"] = df["UTCDate"] + " " + df["UTCTime"]
    df = df.drop(labels = ["UTCDate", "UTCTime"], axis = 1);
    df["time"] = df["time"].map(lambda dt: datetime.strptime(dt, '%Y.%m.%d %H:%M:%S'))

    df.sort_values(by = "time", inplace = True)
    df.reset_index(drop = True, inplace = True)

    first_time = df["time"].iloc[0]
    df["time"] = (df["time"] - first_time).map(lambda delta: int(delta.total_seconds()))    
    return df    

def factor_players(df):
    players = pd.concat([df['White'], df['Black']])
    codes, uniques = pd.factorize(players)
    return codes, uniques    

class PlayerStatistics:
    
    def __init__(self, filename):
        print("Reading CSV...")
        df = pd.read_csv(filename, usecols = ["Event", "Black", "White","BlackElo", "WhiteElo", "UTCDate", "UTCTime", "Result", "TimeControl"])
        df = clean_dataframe(df)
    
        print("Processing time...")
        df = sort_by_time(df)
        
        print("Factoring players...")
        codes, uniques = factor_players(df)
        
        self._players = uniques
    
        self._name_to_code = {uniques[code]:code for code in codes}    
        self._code_to_name = {code:uniques[code] for code in codes}
        
        self._final_elo = defaultdict(dict)

        self._games_played = defaultdict(lambda: defaultdict(int))
        
        self._outcomes = {}
        self._outcomes['White'] = df[df['Result'] == '1-0'].count()
        self._outcomes['Black'] = df[df['Result'] == '0-1'].count()
        self._outcomes['Draw'] = df[df['Result'] == '1/2-1/2'].count()
        
        def game_type(event):
            if "Bullet" in event:
                return "Bullet"
            if "Blitz" in event:
                return "Blitz"
            if "Rapid" in event:
                return "Rapid"
            if "Classical" in event:
                return "Classical"

            
        self._games_played_by_type = {
            "Bullet": 0,
            "Blitz": 0,
            "Rapid": 0,
            "Classical": 0,
            "None": 0,
        }
        
        print("Iterating rows...")
        for row in df.itertuples():
            
            type = game_type(row.Event)
            
            self._final_elo[row.White][type] = int(row.WhiteElo)
            self._final_elo[row.Black][type] = int(row.BlackElo)
            
            self._games_played[row.White][type] += 1
            self._games_played[row.Black][type] += 1
            
            self._games_played_by_type[str(type)] += 1
        
        print("Statistics ready!")

    
    def username_from_code(self, id):
        return self._code_to_name(id)
    
    def code_from_username(self, username):
        return self._name_to_code(username)
    
    def final_elo(self, username):
        return self._final_elo[username]
    
    def games_played(self, username):
        return self._games_played[username]
    
    def games_played_type(self, type):
        return self._games_played_by_type[type]
    
    def outcomes(self):
        return self._outcomes
    
    def players(self):
        return list(self._players)
