import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.backend.model.mock import MockSkill
from app.backend.model.skill import InvalidInput
from app.backend.model.tgl import TGLDeepSkill

# setup loggers
logging.config.fileConfig('app/backend/logging.conf', disable_existing_loggers=False)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

skill_system = TGLDeepSkill()

@app.get("/predict")
async def predict(white: str, black: str, min: int = 10, inc: int = 0):
    time_control = (min, inc)
    try:
        prediction = skill_system.predict(white, black, time_control)    
        return prediction
    except InvalidInput as e:
        raise HTTPException(status_code=400, detail=e.invalid)

@app.get("/")
async def root():
    return {"message": "Hello World"}
