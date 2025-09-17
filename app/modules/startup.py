# app/modules/startup.py

import os
import json
import torch
from app.config import CONFIG, PLAYER_TEMPLATES, CREATURE_TEMPLATES, ACTION_NAMES
from app.modules.creature_manager import create_creature

PLAYERS_DIR = "players"

def simulate_backend_fetch_players():
  """
  Simulated call to the backend.
  Always returns False for now.
  TODO: Replace with real backend call.
  """
  return False

def bootstrap_players():
  """Create initial players and creatures if backend has no data."""
  if not os.path.exists(PLAYERS_DIR):
    os.makedirs(PLAYERS_DIR)
  if not os.path.exists(CONFIG["checkpoint_dir"]):
    os.makedirs(CONFIG["checkpoint_dir"])

  # Simulated backend check
  if simulate_backend_fetch_players():
    print("✅ Players loaded from backend.")
    return

  print("⚠️ Backend unavailable. Bootstrapping from templates...")

  for player_template in PLAYER_TEMPLATES:
    player_id = PLAYER_TEMPLATES[player_template]["id"]
    player_name = PLAYER_TEMPLATES[player_template]["name"]

    creatures = []
    for creature_info in PLAYER_TEMPLATES[player_template]["creatures"]:
      template_key = creature_info["name"]  # 'A' or 'B'
      creature = create_creature(
        template_key=template_key,
        owner=player_name
      )

      # Build checkpoint path
      checkpoint_path = os.path.join(
        CONFIG["checkpoint_dir"], f"{creature.creature_id}.pt"
      )

      # Save initial checkpoint in the same format training_loop expects
      torch.save({
        "model_state_dict": creature.nn_model.state_dict(),
        "optimizer_state_dict": None,  # or init with torch.optim state
        "activations_history": []
      }, checkpoint_path)

      # Add creature entry for player.json
      creature_data = creature.to_dict()
      creature_data["nn_checkpoint"] = checkpoint_path
      creatures.append(creature_data)

    # Save player.json
    player_data = {
      "id": player_id,
      "name": player_name,
      "creatures": creatures
    }
    filepath = os.path.join(PLAYERS_DIR, f"{player_name.lower()}.json")
    with open(filepath, "w") as f:
      json.dump(player_data, f, indent=2)

    print(f"📝 Created {filepath}")
