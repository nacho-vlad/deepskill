from model.skill import SkillRatingSystem

class MockSkill(SkillRatingSystem):
    
    def __init__(self):
        self.matchups = {
            ("John", "Alice"): (0.7, 0.2, 0.1), 
            ("Alice", "John"): (0.3, 0.6, 0.1),
            ("John", "Bobby"): (0.55, 0.25, 0.2),
            ("Bobby", "John"): (0.35, 0.65, 0.1),
            ("Alice", "Bobby"): (0.0, 1.0, 0.0),
            ("Bobby", "Alice"): (1.0, 0.0, 0.0),
        }
    
    def predict(self, white: str, black: str, tc: (int, int)):
        
        if (white, black) not in self.matchups:
            return None
        
        results = self.matchups[(white, black)]
        return {
            'white': results[0],
            'black': results[1],
            'draw': results[2],
        }