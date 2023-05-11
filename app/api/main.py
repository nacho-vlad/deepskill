from fastapi import FastAPI
from model.mock import MockSkill

app = FastAPI()

skill_system = MockSkill()

@app.get("/predict")
async def predict(white: str, black: str, min: int = 10, inc: int = 0):
    time_control = (min, inc)
    return skill_system.predict(white, black, time_control)

@app.get("/")
async def root():
    return {"message": "Hello World"}
