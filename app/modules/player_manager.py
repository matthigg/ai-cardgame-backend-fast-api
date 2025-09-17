# app/modules/player_manager.py
import os
from typing import Dict
from app.modules.player import Player, load_player, save_player
from app.config import CREATURE_TEMPLATES, PLAYER_TEMPLATES

# Active player registry
# Key: "name_id"
_active_players: Dict[str, Player] = {}

def _make_key(name: str, pid: int) -> str:
  return f"{name}_{pid}"

def get_active_player(name: str, pid: int) -> Player | None:
  return _active_players.get(_make_key(name, pid))

def add_active_player(name: str, pid: int, players_dir="players") -> Player | None:
  key = _make_key(name, pid)
  if key in _active_players:
    return _active_players[key]  # already active

  os.makedirs(players_dir, exist_ok=True)
  path = os.path.join(players_dir, f"player_{pid}.json")

  # Init all creatures from templates
  creatures, _ = init_creatures(CREATURE_TEMPLATES)

  if os.path.exists(path):
    # Load existing player file
    player = load_player(path, creatures)
  else:
    # Create new player from template
    template = PLAYER_TEMPLATES.get(pid, {"name": name, "creatures": []})
    player = Player(template["name"], pid)

    for cdata in template.get("creatures", []):
      ckey = cdata["template"]  # template key
      cid = cdata["id"]         # unique creature ID
      if ckey in CREATURE_TEMPLATES:
        # create persistent creature per player with correct ID
        creature = create_creature(ckey, owner=player.name, creature_id=cid)
        add_active_creature(creature)
        player.add_creature(creature)

    # Save the newly created player
    save_player(player)

  _active_players[key] = player
  return player


def remove_active_player(name: str, pid: int):
  _active_players.pop(_make_key(name, pid), None)

def list_active_players():
  return list(_active_players.keys())
