import json
import os
import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.config import CONFIG
from app.modules.training_loop import training_loop
from app.modules.utils import get_checkpoint_path

router = APIRouter()

class TrainRequest(BaseModel):
  player_name_A: str
  player_id_A: int
  creature_name_A: str
  creature_id_A: int
  player_name_B: str
  player_id_B: int
  creature_name_B: str
  creature_id_B: int

@router.post("/train")
def train_endpoint(request: TrainRequest):
  """Run full training loop and save checkpoints, returning final summary."""
  result = training_loop(
    request.player_name_A, request.player_id_A, request.creature_name_A, request.creature_id_A,
    request.player_name_B, request.player_id_B, request.creature_name_B, request.creature_id_B
  )
  return {"status": "completed", "summary": result.get("summary")}

@router.get("/summary")
def get_summary():
  """Return summary JSON if available."""
  filename = os.path.join(CONFIG['log_dir'], 'summary.json')
  if os.path.exists(filename):
    with open(filename, 'r') as f:
      return json.load(f)
  return {"error": "Summary not available yet"}

@router.post("/nn-graph")
def nn_graph(request: TrainRequest):
  """
  Return weights, biases, and normalized activations_history
  for Alice's Bear and Bob's Snake.
  """

  creatures = [
    (request.player_name_A, request.player_id_A, request.creature_name_A, request.creature_id_A),
    (request.player_name_B, request.player_id_B, request.creature_name_B, request.creature_id_B),
  ]

  result = {}
  for player_name, player_id, creature_name, creature_id in creatures:
    checkpoint_path = get_checkpoint_path(player_name, player_id, creature_name, creature_id)
    if not os.path.exists(checkpoint_path):
      result[creature_name] = {"error": f"Checkpoint not found for {creature_name}"}
      continue

    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint.get('model_state_dict', {})
    activations_history = checkpoint.get('activations_history', [])

    weights, biases = [], []
    for key in sorted(state_dict.keys()):
      tensor = state_dict[key]
      if 'weight' in key:
        weights.append(tensor.tolist())
      elif 'bias' in key:
        biases.append(tensor.tolist())

    normalized_activations = []
    for epoch_entry in activations_history:
      epoch_layers = []
      for layer in epoch_entry['layers']:
        flat_layer = [float(neuron) for neuron in layer if isinstance(neuron, (int, float))]
        epoch_layers.append(flat_layer)
      normalized_activations.append({
        "name": epoch_entry['name'],
        "epoch": epoch_entry['epoch'],
        "layers": epoch_layers
      })

    result[creature_name] = {
      "weights": weights,
      "biases": biases,
      "activations_history": normalized_activations
    }

  return result
