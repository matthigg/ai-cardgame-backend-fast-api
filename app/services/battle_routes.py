import json
import os
import torch
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from app.config import CONFIG, CREATURE_TEMPLATES
from app.modules.training_loop import training_loop
from app.modules.utils import create_checkpoint_paths_by_name
from app.modules.player_manager import get_active_player
from app.modules.creature_manager import _active_creatures, init_creatures

router = APIRouter()

@router.get("/train/player")
def train_player_creature(
  player_name: str = Query(..., description="Name of the player"),
  pid: int = Query(..., description="Player ID"),
  creature_name: str = Query(..., description="Name of the player's creature"),
  opponent_name: str = Query(None, description="Optional opponent creature name (player or NPC)"),
  epochs: int = Query(CONFIG['epoch_batch_size'], description="Number of training epochs")
):
  """
  Train a specific player's creature against an optional opponent.
  If opponent is not provided, use a default NPC template.
  """
  from app.modules.player_manager import get_active_player, add_active_player, list_active_players

  # --- Basic validation ---
  if not creature_name or creature_name.strip() == "":
    return JSONResponse({"error": "creature_name must be provided"}, status_code=400)

  # Debug/log current active players (prints to server console)
  print(f"[TRAIN] request player_name={player_name} pid={pid} creature_name={creature_name} opponent_name={opponent_name} epochs={epochs}")
  print("[TRAIN] active players before lookup:", list_active_players())

  # Attempt to find active player
  player = get_active_player(player_name, pid)

  # If not active, try to load from disk (this will create the player record if missing)
  if not player:
    print(f"[TRAIN] player {player_name} ({pid}) not active - attempting to load/create from disk...")
    player = add_active_player(player_name, pid)
    if not player:
      print("[TRAIN] failed to load/create player; active players now:", list_active_players())
      return JSONResponse({"error": f"Player {player_name} ({pid}) not found or could not be loaded"}, status_code=404)

  # Confirm player loaded
  print(f"[TRAIN] using player: {player.name} ({player.id}), creatures: {[c.name for c in player.creatures]}")

  # Lookup player's creature
  player_creature = next((c for c in player.creatures if c.name == creature_name), None)
  if not player_creature:
    return JSONResponse({"error": f"Creature {creature_name} not found for player {player_name}"}, status_code=404)

  # Determine opponent creature
  opponent_creature = None
  if opponent_name:
    # Check if opponent is an active player's creature
    for c in _active_creatures.values():
      if c.name == opponent_name:
        opponent_creature = c
        break
    # If not found, check templates for NPC
    if not opponent_creature:
      templates, _ = init_creatures({opponent_name: {}})
      opponent_creature = templates.get(opponent_name, None)

  # If no opponent found, pick a default template
  if opponent_creature is None:
    templates, _ = init_creatures(CREATURE_TEMPLATES)
    default_name = list(templates.keys())[0]
    opponent_creature = templates[default_name]

  # Call training loop
  result = training_loop(player_creature, opponent_creature, epochs=epochs)

  return {
    "status": "completed",
    "summary": result.get("summary"),
    "activations": result.get("activations")
  }



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

  weights, biases = [], []

  for key in sorted(state_dict.keys()):
    tensor = state_dict[key]
    if 'weight' in key:
      weights.append(tensor.tolist())
    elif 'bias' in key:
      biases.append(tensor.tolist())

  # Normalize activations
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

  return {
    "name": creature_name,
    "weights": weights,
    "biases": biases,
    "activations_history": normalized_activations
  }
