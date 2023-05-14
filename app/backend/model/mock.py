from app.backend.model.skill import SkillRatingSystem, InvalidInput

class MockSkill(SkillRatingSystem):
    
    players = ["John", "Alice", "Bobby"]
    
    def __init__(self):
        self.matchups = {
            ("John", "Alice"): (0.7, 0.2, 0.1), 
            ("Alice", "John"): (0.3, 0.6, 0.1),
            ("John", "Bobby"): (0.55, 0.25, 0.2),
            ("Bobby", "John"): (0.35, 0.65, 0.1),
            ("Alice", "Bobby"): (0.0, 1.0, 0.0),
            ("Bobby", "Alice"): (1.0, 0.0, 0.0),
        }
        
    def validate(self, white: str, black: str, tc: (int, int)):
        
        error = False
        invalid = []
        
        if white not in self.players:
            error = True
            invalid.append('white')
            pass
        if black not in self.players:
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
        
        results = self.matchups[(white, black)]
        return {
            'white': results[0],
            'black': results[1],
            'draw': results[2],
        }