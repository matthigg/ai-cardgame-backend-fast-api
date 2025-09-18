# app/modules/player_factory.py
import os
import json
import torch
from app.modules.creature_manager import Creature, build_nn_for_creature
from app.modules.player import Player
from app.modules.utils import get_checkpoint_path, get_player_json_path
from app.config import CREATURE_TEMPLATES

def create_player(name: str, creature_keys: list):
  """Create a Player instance, its creatures, and checkpoint files."""
  player = Player(name)
  for idx, key in enumerate(creature_keys):
    template = CREATURE_TEMPLATES[key]

    # Unique creature ID: player.id * 10 + idx + 1
    creature_id = player.id * 10 + (idx + 1)
    nn_model = build_nn_for_creature(template)

    # Save initial checkpoint
    checkpoint_path = get_checkpoint_path(player.name, player.id, template['name'], creature_id)
    torch.save({
      "model_state_dict": nn_model.state_dict(),
      "optimizer_state_dict": None,
      "activations_history": []
    }, checkpoint_path)

    # Create Creature instance and add checkpoint path
    creature = Creature(template['name'], player.name, nn_model, template, creature_id)
    creature_data = creature.to_dict()
    creature_data['nn_checkpoint'] = checkpoint_path

    player.add_creature(creature)

  # Save player.json
  player_json_path = get_player_json_path(player.name, player.id)
  with open(player_json_path, "w") as f:
    json.dump({
      "id": player.id,
      "name": player.name,
      "creatures": [c.to_dict() for c in player.creatures]
    }, f, indent=2)

  return player, player_json_path
