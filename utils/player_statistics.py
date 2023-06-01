import pandas as pd
from collections import defaultdict
from csv_to_input import sort_by_time, factor_players, clean_dataframe

class PlayerStatistics():
    
    def __init__(filename):
        print("Reading CSV...")
        df = pd.read_csv(filename, usecols = ["Event", "Black", "White","BlackElo", "WhiteElo", "UTCDate", "UTCTime", "Result", "TimeControl"])
        df = clean_dataframe(df)
    
        print("Processing time...")
        df = sort_by_time(df)
        
        print("Factoring players...")
        codes, uniques = factor_players(df)
    
        self.name_to_code = {uniques[code]:code for code in codes}    
        self.code_to_name = {code:uniques[code] for code in codes}
        
        self.final_elo = defaultdict(dict)

        self.games_played = defaultdict(lambda: defaultdict(int))
        
        self.outcomes = {}
        self.outcomes['White'] = df[df['Result'] == '1-0'].count()
        self.outcomes['Black'] = df[df['Result'] == '0-1'].count()
        self.outcomes['Draw'] = df[df['Result'] == '1/2-1/2'].count()
        
        def type(event):
            if "Bullet" in event:
                return "Bullet"
            if "Blitz" in event:
                return "Blitz"
            if "Rapid" in event:
                return "Rapid"
            if "Classical" in event:
                return "Classical"
            
        self.games_played_by_type = {
            "Bullet": 0,
            "Blitz": 0,
            "Rapid": 0,
            "Classical" 0,
        }
        
        print("Iterating rows...")
        for row in df.itertuples():
            
            type = type(row.Event)
            
            self.final_elo[row.White][type] = row.WhiteElo
            self.final_elo[row.Black][type] = row.BlackElo
            
            self.games_played[row.White][type] += 1
            self.games_played[row.Black][type] += 1
            
            self.games_played_by_type[type] += 1
        
        print("Statistics ready!")

    
    def username_from_code(self, id):
        return self.code_to_name(id)
    
    def code_from_username(self, username):
        return self.name_to_code(username)
    
    def final_elo(self, username):
        return self.final_elo(username)
    
    def games_played(self, username):
        return self.games_played[username]
    
    def games_played_type(self, type):
        return self.games_played_by_type[type]
    
    def outcomes(self):
        return self.outcomes
