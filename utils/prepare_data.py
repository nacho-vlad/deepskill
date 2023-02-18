import argparse
from datetime import datetime
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description='Prepare input data for a specific month.')
parser.add_argument('month', help='the months to prepare')

args = parser.parse_args()

file_path = Path(__file__).parent
file_path = file_path.parent / "data" / "processed"

def prepare_tensor(df):
    
    lst = df.values.tolist()
    print(lst)
    
    def convert_result_to_int(result):
        return {
            "0-1": 0,
            "1-0": 1,
            "1/2-1/2": 2,
        }[result]
    

for file in file_path.glob('*' + args.month + '*'):
    df = pd.read_csv(file, usecols = ["Black", "White","UTCDate", "UTCTime", "Result", "TimeControl"])
    
    df["time"] = df["UTCDate"] + " " + df["UTCTime"]
    df = df.drop(labels = ["UTCDate", "UTCTime"], axis = 1);
    df["time"] = df["time"].map(lambda dt: datetime.strptime(dt, '%Y.%m.%d %H:%M:%S'))
    
    df.sort_values(by = "time", inplace = True)
    df.reset_index(drop = True, inplace = True)

    prepare_tensor(df[["Result", "TimeControl"]])
    df = df.drop(labels = ["Result", "TimeControl"], axis = 1)
    
    players = pd.concat([df['White'], df['Black']])
    codes, uniques = pd.factorize(players)
    matches = len(codes) // 2
    
    df = df.drop(labels = ["Black", "White"], axis = 1)
    
    df["src"] = codes[:matches]
    df["dst"] = codes[matches:]
    
    first_time = df["time"].iloc[0]
    df["time"] = (df["time"] - first_time).map(lambda delta: int(delta.total_seconds()))
    
    
    p50 = len(df) // 2
    p75 = len(df) * 3 // 4
    p100 = len(df)

    df["ext_roll"] = 0
    df["ext_roll"].iloc[p50:p75] = 1
    df["ext_roll"].iloc[p75:p100] = 2
    
    print(df)
    
    
    