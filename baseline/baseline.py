import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--month', type=str, help='month name')
args=parser.parse_args()

import os
from pathlib import Path

file = Path(__file__)
os.chdir(file.parent)

month = args.month
DATA_PATH = f'../data/processed/lichess_db_standard_rated_{month}.csv'

from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

games = pd.read_csv(DATA_PATH)
print(games.head())

games = games[(games.WhiteElo != '?') & (games.BlackElo != '?')]

X = games[['WhiteElo', 'BlackElo']].astype(int)
y = games['Result']

le = LabelEncoder()
le.fit(y)
y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)


clf = XGBClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_test = le.inverse_transform(y_test)
y_pred = le.inverse_transform(y_pred)

print(classification_report(y_test, y_pred, digits=4, zero_division = 0))
