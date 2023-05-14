import logging
import os

from app.backend.model.skill import SkillRatingSystem
from tgl.model import TemporalGraphModel

logger = logging.getLogger(__name__)


class TGLDeepSkill(SkillRatingSystem):
    
    def __init__(self):
        logger.info(os.getcwd())
        pass
    
    def predict(self, white: str, black: str, tc: (int, int)):
        pass
        