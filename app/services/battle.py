import json
import os
import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.config import CONFIG, CREATURES
from app.modules.training_loop_stream import training_loop  # <- updated synchronous version
from app.modules.utils import create_checkpoint_paths_by_name

router = APIRouter()

@router.get("/train")
def train_endpoint():
  """Run full training loop and save checkpoints, returning final summary."""
  result = training_loop()
  return {"status": "completed", "summary": result.get("summary")}

@router.get("/summary")
def get_summary():
  """Return summary JSON if available."""
  filename = os.path.join(CONFIG['log_dir'], 'summary.json')
  if os.path.exists(filename):
    with open(filename, 'r') as f:
      return json.load(f)
  return {"error": "Summary not available yet"}

@router.get("/nn-graph/{creature_name}")
def nn_graph(creature_name: str):
  """Return weights, biases, and normalized activations_history for a creature."""

  A_path, B_path = create_checkpoint_paths_by_name('A', 'B')
  path_map = {'A': A_path, 'B': B_path}

  if creature_name not in path_map:
    return JSONResponse({"error": "Invalid creature name"}, status_code=400)

  checkpoint_path = path_map[creature_name]
  if not os.path.exists(checkpoint_path):
    return JSONResponse({"error": "Checkpoint not found"}, status_code=404)

  checkpoint = torch.load(checkpoint_path)
  state_dict = checkpoint.get('model_state_dict', {})
  activations_history = checkpoint.get('activations_history', [])

  # print('activations_history: ', activations_history)

  weights, biases = [], []

  for key in sorted(state_dict.keys()):
    tensor = state_dict[key]
    if 'weight' in key:
      weights.append(tensor.tolist())
    elif 'bias' in key:
      biases.append(tensor.tolist())

  # Normalize activations per layer, respecting the new structure
  normalized_activations = []
  for epoch_entry in activations_history:
      
      # print('epoch_entry: ', epoch_entry)

      epoch_layers = []
      for layer in epoch_entry['layers']:
          # Ensure each neuron is a float (ignore extra display neurons)
          flat_layer = [float(neuron) for neuron in layer if isinstance(neuron, (int, float))]
          epoch_layers.append(flat_layer)
      normalized_activations.append({
          "name": epoch_entry['name'],
          "epoch": epoch_entry['epoch'],
          "layers": epoch_layers
      })


  return {
    "weights": weights,
    "biases": biases,
    "activations_history": normalized_activations
  }
