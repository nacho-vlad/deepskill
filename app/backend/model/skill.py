class InvalidInput(Exception):
    def __init__(self, invalid):
        self.invalid = invalid

class SkillRatingSystem:
    
    def validate(self, white: str, black: str, tc: (int, int)):
        pass
    
    def predict(self, white: str, black: str, tc: (int, int)):
        pass
