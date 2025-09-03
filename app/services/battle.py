from fastapi import APIRouter
from app.config import ACTION_NAMES, CONFIG
from app.modules.battle_simulation import simulate_battle
from app.modules.creature import Creature
from app.modules.neural_network import NeuralNetwork
from app.modules.training_loop import training_loop

router = APIRouter()

@router.get("/train")
def train_endpoint():
  training_loop()
  return {"status": "Training completed"}
