# app/modules/player_manager.py
import os
from typing import Dict
from app.modules.player import Player, load_player, save_player
from app.modules.creature_manager import init_creatures, load_creature, add_active_creature, create_creature
from app.config import CREATURE_TEMPLATES, PLAYER_TEMPLATES

# Active player registry
# Key: "name_id"
_active_players: Dict[str, Player] = {}

def _make_key(name: str, pid: int) -> str:
  return f"{name}_{pid}"

def get_active_player(name: str, pid: int | None = None) -> Player | None:

  print('name: ', name)
  print('pid: ', pid)
  print('_active_players: ', _active_players)
  
  if pid is None:
    # fallback: return first matching active player
    for key, player in _active_players.items():
      if player.name == name:
        return player
    return None
  return _active_players.get(_make_key(name, pid))


def add_active_player(name: str, pid: int, players_dir="players") -> Player | None:
  key = _make_key(name, pid)
  if key in _active_players:
    return _active_players[key]  # already active

  os.makedirs(players_dir, exist_ok=True)
  path = os.path.join(players_dir, f"player_{pid}.json")

  # Init creatures so we can attach them to player
  creatures, _ = init_creatures(CREATURE_TEMPLATES)

  if os.path.exists(path):
    # Load existing player file
    player = load_player(path, creatures)
  else:
    # Create new player from template or blank
    template = PLAYER_TEMPLATES.get(name, {"name": name, "creatures": []})
    player = Player(template["name"], pid)
    for ckey in template.get("creatures", []):
      if ckey in CREATURE_TEMPLATES:
        # Create persistent creature per player
        creature = create_creature(ckey, owner=player.name)
        add_active_creature(creature)
        player.add_creature(creature)
    save_player(player)

  _active_players[key] = player
  return player

def remove_active_player(name: str, pid: int):
  _active_players.pop(_make_key(name, pid), None)

def list_active_players():
  return list(_active_players.keys())
