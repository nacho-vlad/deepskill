import logging
import os

from app.backend.model.skill import SkillRatingSystem, InvalidInput
from tgl.model import TemporalGraphModel
from utils.player_statistics import PlayerStatistics

logger = logging.getLogger(__name__)

DATA = "tgl/DATA/LICHESS-2013-06"
CONFIG = "tgl/config/TGN_PRODUCTION.yml"
STORED_MODEL = "tgl/models/1685627744.312774.pkl"

PLAYER_DATA = "data/processed/lichess_db_standard_rated_2013-06.csv"

class TGLDeepSkill(SkillRatingSystem):
    
    def __init__(self):
        logger.info(os.getcwd())
        self.model = TemporalGraphModel(DATA, CONFIG, STORED_MODEL, supervised = True)
        self.player_stats = PlayerStatistics(PLAYER_DATA)
    
    def validate(self, white: str, black: str, tc: (int, int)):

        error = False
        invalid = []
        
        if white not in self.player_stats.players():
            error = True
            invalid.append('white')
            pass
        if black not in self.player_stats.players():
            error = True
            invalid.append('black')
            pass
        
        min = tc[0]
        inc = tc[0]
        
        if min is None or min < 1:
            error = True
            invalid.append('min')
        
        if inc is None or inc < 0:
            error = True
            invalid.append('inc')
        
        if error:
            raise InvalidInput(invalid)
    
    def predict(self, white: str, black: str, tc: (int, int)):
        
        self.validate(white, black, tc)
        white_node = self.player_stats.code_from_username(white)
        black_node = self.player_stats.code_from_username(black)
        pred = self.model.get_prediction(white_node, black_node, tc)
        
        return {
            'white': pred[0],
            'black': pred[1],
            'draw': pred[2],
        }
        