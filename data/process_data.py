import chess.pgn
import os
import csv
from pathlib import Path

os.chdir(os.path.abspath(os.path.dirname(__file__) ))

directory = "raw"

files = list(
    filter(
        lambda file: file.endswith("pgn"),
        os.listdir("raw")
    ))

for filename in files:
    if filename.endswith("lichess_db_standard_rated_2022-11.pgn"):
        continue
        
    pgn = open(f"raw/{filename}")
    
    csv_filename = Path(filename).stem + ".csv"
    csv_file = open(f"processed/{csv_filename}", "w", newline="")
    
    fieldnames = ["Event", "White", "Black", "Result", "BlackElo", "WhiteElo", "BlackRatingDiff", "WhiteRatingDiff", "Opening", "Termination", "TimeControl", "UTCDate", "UTCTime"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    
    writer.writeheader()
    
    game_count = 1
    game = chess.pgn.read_game(pgn)
    while game:
        print(f"Game number: {game_count}")
        writer.writerow(game.headers)
        game = chess.pgn.read_game(pgn)
        game_count += 1
    
    close(pgn)
    close(csv_file)
    
