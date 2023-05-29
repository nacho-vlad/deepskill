import argparse
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import torch as pt
from pathlib import Path

parser = argparse.ArgumentParser(description='Prepare input data for a specific month.')
parser.add_argument('month', help='the months to prepare')

args = parser.parse_args()

file_path = Path(__file__).parent
csv_path = file_path.parent / "data" / "processed"
save_path = file_path.parent / "tgl" / "DATA" / "LICHESS-" / args.month
save_path.mkdir(parents = True, exist_ok = True)

"""
Dict that maps player code to:
Name, Elo, MatchCount
"""
def compute_player_statistics(df, codes, players):
    
    df = df[df.WhiteElo != '?']
    df = df[df.BlackElo != '?']
    
    name_to_code = {name:code for (name, code) in enumerate(players)}    
    
    grouped_white = df.groupby('White')
    grouped_black = df.groupby('Black')
    
    match_count = grouped_white.size().add(grouped_black.size(), fill_value=0).astype(int)
    
    elo = pd.concat([grouped_white['WhiteElo'].last(), grouped_black['BlackElo'].last()], axis = 1).fillna(0)
    elo = elo.astype(int)
    elo = elo[['WhiteElo', 'BlackElo']].apply(max, axis = 1)
    
    code_to_stats = []
    for i in range(len(players)):
        stats = {}
        name = players[i]
        
        stats['name'] = name
        
        try:
            stats['matches'] = match_count[name]
            stats['elo'] = elo[name]
        except:
            stats['matches'] = 0
            stats['elo'] = 0
        code_to_stats.append(stats)
    
    with open(save_path / 'stats.pkl', 'wb') as f:
        pickle.dump(code_to_stats, f)
    
    
def prepare_tensor(df):
    lst = df.values.tolist()
    
    def convert_result(result):
        return {
            "1-0": 0,
            "0-1": 1,
            "1/2-1/2": 2,
        }[result]
    
    def time_control_normalized(tc):
        if tc == '-':
            return 2.0
        return int(tc.split("+")[0]) / 1200
    
    def time_control_inc_normalized(tc):
        if tc == '-':
            return 0.0
        return int(tc.split("+")[1]) / 10
    
    lst = list(map(lambda l: 
        [convert_result(l[0]), 
         time_control_normalized(l[1]), 
         time_control_inc_normalized(l[1])],
        lst))
    
    arr = np.array(lst)
    
    time_control = pt.from_numpy(arr[:, 1:])
    one_hot_results = pt.nn.functional.one_hot(pt.from_numpy(arr[:, 0]).to(pt.int64))
    tensor = pt.cat((one_hot_results, time_control), 1)
    pt.save(tensor, save_path / "edge_features.pt")
    print(tensor.size())

def prepare_input(file):
    print("Reading CSV...")
    df = pd.read_csv(file, usecols = ["Event", "Black", "White","BlackElo", "WhiteElo", "UTCDate", "UTCTime", "Result", "TimeControl"])
    
    print("Processing time...")
    df["time"] = df["UTCDate"] + " " + df["UTCTime"]
    df = df.drop(labels = ["UTCDate", "UTCTime"], axis = 1);
    df["time"] = df["time"].map(lambda dt: datetime.strptime(dt, '%Y.%m.%d %H:%M:%S'))

    first_time = df["time"].iloc[0]
    df["time"] = (df["time"] - first_time).map(lambda delta: int(delta.total_seconds()))
    
    # should already be sorted by time
    #df.sort_values(by = "time", inplace = True)
    #df.reset_index(drop = True, inplace = True)
    
    print("Preparing tensor...")
    df = df[df['Result'].isin(["1-0", "0-1", "1/2-1/2"])]
    prepare_tensor(df[["Result", "TimeControl"]])
    df = df.drop(labels = ["Result", "TimeControl"], axis = 1)
    
    print("Factoring players...")
    players = pd.concat([df['White'], df['Black']])
    codes, uniques = pd.factorize(players)
    matches = len(codes) // 2

    print("Computing player statistics...")
    compute_player_statistics(df, codes, uniques)
    df = df.drop(labels = ["Event", "Black", "White", "BlackElo", "WhiteElo"], axis = 1)
    
    df["src"] = codes[:matches]
    df["dst"] = codes[matches:]
        
    p50 = len(df) // 2
    p75 = len(df) * 3 // 4
    p100 = len(df)

    df["ext_roll"] = 0
    df["ext_roll"].iloc[p50:p75] = 1
    df["ext_roll"].iloc[p75:p100] = 2
    
    print("Saving to CSV")
    df.to_csv(save_path / "edges.csv")
    print(df)
    
file = list(csv_path.glob('*' + args.month + '*'))[0]
print(file)
prepare_input(file)
    
    
    