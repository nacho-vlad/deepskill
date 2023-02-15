import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description='Prepare input data for a specific month.')
parser.add_argument('month', help='the months to prepare')

args = parser.parse_args()

file_path = Path(__file__).parent
file_path = file_path.parent / "data" / "processed"


for file in file_path.glob('*' + args.month + '*'):
    df = pd.read_csv(file, usecols = ["Black", "White"])
    players = pd.concat([df['White'], df['Black']])
    print(players)
    codes, uniques = pd.factorize(players)
    print(codes)
    