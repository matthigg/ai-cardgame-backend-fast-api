from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.config import ACTION_NAMES, CONFIG
from app.modules.battle_simulation import simulate_battle
from app.modules.creature import Creature
from app.modules.neural_network import NeuralNetwork
from app.modules.training_loop import training_loop
from app.modules.training_loop_stream import training_loop_stream


router = APIRouter()

@router.get("/train")
def train_endpoint():
  training_loop()
  return {"status": "Training completed"}

@router.get("/training-stream")
async def training_stream():
  return StreamingResponse(training_loop_stream(), media_type="text/event-stream")
