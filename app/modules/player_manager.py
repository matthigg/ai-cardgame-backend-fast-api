# app/modules/player_manager.py
import os
import itertools
import json
import torch
from app.modules.creature_manager import Creature, build_nn_for_creature
from app.modules.utils import get_checkpoint_path, get_npc_json_path, get_player_json_path
from app.config import CREATURE_TEMPLATES, GENERATED_DIR, NPCS_DIR, PLAYERS_DIR

# global unique player id counter
_player_id_counter = itertools.count(1)

class Player:
  def __init__(self, name, player_id=None):
    self.id = player_id or next(_player_id_counter)
    self.name = name
    self.creatures = []  # list of Creature instances

  def add_creature(self, creature):
    self.creatures.append(creature)

  def reset(self):
    for c in self.creatures:
      c.reset()

  def to_dict(self):
    return {
      "id": self.id,
      "name": self.name,
      "creatures": [c.name for c in self.creatures]
    }

  @classmethod
  def from_dict(cls, data, all_creatures):
    player = cls(data['name'], data['id'])
    for cname in data['creatures']:
      if cname in all_creatures:
        player.add_creature(all_creatures[cname])
    return player


def save_player(player: Player):
  os.makedirs(PLAYERS_DIR, exist_ok=True)
  path = os.path.join(PLAYERS_DIR, f"player_{player.id}.json")
  with open(path, "w") as f:
    json.dump(player.to_dict(), f, indent=2)
  return path


def load_player(path, all_creatures):
  with open(path, "r") as f:
    data = json.load(f)
  return Player.from_dict(data, all_creatures)


def create_player(name: str, creature_keys: list):
  """Create a Player instance, its creatures, and checkpoint files."""
  player = Player(name)

  for idx, key in enumerate(creature_keys):
    template = CREATURE_TEMPLATES[key]

    # Unique creature ID: player.id * 10 + idx + 1
    creature_id = player.id * 10 + (idx + 1)
    nn_model = build_nn_for_creature(template)

    # Create optimizer for the creature
    optimizer = torch.optim.Adam(
      nn_model.parameters(),
      lr=template.get('nn_config', {}).get('learning_rate', 0.001)
    )

    # Save initial checkpoint with valid optimizer state
    checkpoint_path = get_checkpoint_path(player.name, player.id, template['name'], creature_id)
    torch.save({
      "model_state_dict": nn_model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "activations_history": []
    }, checkpoint_path)

    # Create Creature instance and add checkpoint path
    creature = Creature(template['name'], player.name, nn_model, template, creature_id)
    creature_data = creature.to_dict()
    creature_data['nn_checkpoint'] = checkpoint_path
    creature_data['creature_template_id'] = template.get('creature_template_id')  # <-- fixed

    player.add_creature(creature)

  # Save player.json
  player_json_path = get_player_json_path(player.name, player.id)
  with open(player_json_path, "w") as f:
    json.dump({
      "id": player.id,
      "name": player.name,
      "creatures": [
        dict(
          c.to_dict(),
          creature_template_id=CREATURE_TEMPLATES[creature_keys[idx]].get('creature_template_id')
        )
        for idx, c in enumerate(player.creatures)
      ]
    }, f, indent=2)

  return player, player_json_path


def create_npc(name: str, creature_keys: list):
  """
  Create an NPC with one or more creatures.
  Stores its JSON in generated/npcs instead of players.
  """
  os.makedirs(os.path.join(GENERATED_DIR, NPCS_DIR), exist_ok=True)

  npc = Player(name)  # reuse Player class as a container

  for idx, key in enumerate(creature_keys):
    template = CREATURE_TEMPLATES[key]

    # Unique creature ID: npc.id * 10 + idx + 1
    creature_id = npc.id * 10 + (idx + 1)
    nn_model = build_nn_for_creature(template)

    optimizer = torch.optim.Adam(
      nn_model.parameters(),
      lr=template.get("nn_config", {}).get("learning_rate", 0.001)
    )

    checkpoint_path = get_checkpoint_path(
      npc.name, npc.id, template["name"], creature_id
    )
    torch.save({
      "model_state_dict": nn_model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "activations_history": []
    }, checkpoint_path)

    creature = Creature(template["name"], npc.name, nn_model, template, creature_id)
    creature_data = creature.to_dict()
    creature_data["nn_checkpoint"] = checkpoint_path
    creature_data["creature_template_id"] = template.get('creature_template_id')  # <-- fixed

    npc.add_creature(creature)

  # Save npc.json
  npc_json_path = get_npc_json_path(npc.name, npc.id)
  with open(npc_json_path, "w") as f:
    json.dump({
      "id": npc.id,
      "name": npc.name,
      "type": "npc",
      "creatures": [
        dict(
          c.to_dict(),
          creature_template_id=CREATURE_TEMPLATES[creature_keys[idx]].get('creature_template_id')
        )
        for idx, c in enumerate(npc.creatures)
      ]
    }, f, indent=2)

  return npc, npc_json_path
