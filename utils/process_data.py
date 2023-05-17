import chess.pgn
import os
import csv
import zstandard
import io
import re
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('pattern')
args = parser.parse_args()

pattern = args.pattern


os.chdir(os.path.abspath(os.path.dirname(__file__) ))
os.chdir("../data")

files = list(
    filter(
        lambda file: file.endswith("pgn.zst"),
        os.listdir("raw")
    ))


for filename in files:
    if re.search(pattern, filename) is None:
        continue
    print(filename)

    dctx = zstandard.ZstdDecompressor()
    compressed = open(f"raw/{filename}", 'rb')
    
    pgn = io.TextIOWrapper(
        io.BufferedReader(
            dctx.stream_reader(
                compressed, 
                closefd = True)),
        encoding='utf-8')
    print(pgn)
        
    csv_filename = filename.rstrip(".png.zst") + ".csv"
    csv_file = open(f"processed/{csv_filename}", "w", newline="")
    
    fieldnames = ["Event", "White", "Black", "Result", "BlackElo", "WhiteElo", "BlackRatingDiff", "WhiteRatingDiff", "Opening", "Termination", "TimeControl", "UTCDate", "UTCTime"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    
    writer.writeheader()
    
    game_count = 1
    game = chess.pgn.read_game(pgn)
    while game:
        if game_count % 1000 == 0:
            print(f"Game number: {game_count}")
        writer.writerow(game.headers)
        game = chess.pgn.read_game(pgn)
        game_count += 1
    
    close(csv_file)
