from fastapi import APIRouter
from app.config import ACTION_NAMES, CONFIG
from app.modules.battle_simulation import simulate_battle
from app.modules.creature import Creature
from app.modules.neural_network import NeuralNetwork
from app.modules.training_loop import training_loop

router = APIRouter()

@router.get("/simulate")
def simulate_example():
    input_size = 4
    output_size = len(ACTION_NAMES)
    nn_A = NeuralNetwork(input_size, CONFIG['hidden_sizes'], output_size)
    nn_B = NeuralNetwork(input_size, CONFIG['hidden_sizes'], output_size)
    config_stats = {'hp': 100, 'max_hp': 100, 'energy': 100, 'max_energy': 100}
    creature_A = Creature("A", nn_A, config_stats)
    creature_B = Creature("B", nn_B, config_stats)
    reward_A, reward_B, log, winner = simulate_battle(creature_A, creature_B, 0, 10, 0.3)
    return {"reward_A": reward_A, "reward_B": reward_B, "winner": winner}

@router.get("/train")
def train_endpoint():
    training_loop()
    return {"status": "Training completed"}
