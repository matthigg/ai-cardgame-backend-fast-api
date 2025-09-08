# app/modules/player.py
import copy
import itertools
from app.modules.creature import create_creature, init_creatures

# global unique player id counter
_player_id_counter = itertools.count(1)

class Player:
  def __init__(self, name, player_id=None):
    self.id = player_id or next(_player_id_counter)
    self.name = name
    self.creatures = []   # list of Creature instances

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

# app/modules/player.py (continued)
import os
import json
from app.config import PLAYER_TEMPLATES, CREATURE_TEMPLATES, CONFIG
from app.modules.creature import init_creatures

def init_players(player_templates=PLAYER_TEMPLATES, creature_templates=CREATURE_TEMPLATES):
  creatures, optimizers = init_creatures(creature_templates)
  players = {}

  for pname, pdata in player_templates.items():
    player = Player(pdata['name'])
    for ckey in pdata['creatures']:
      if ckey in creatures:
        player.add_creature(creatures[ckey])
    players[pname] = player

  return players, creatures, optimizers

# app/modules/player_persistence.py
import os, json
# from app.modules.player import Player

PLAYER_DIR = "players"

def save_player(player: Player):
  os.makedirs(PLAYER_DIR, exist_ok=True)
  path = os.path.join(PLAYER_DIR, f"player_{player.id}.json")
  with open(path, "w") as f:
    json.dump(player.to_dict(), f, indent=2)
  return path

def load_player(path, all_creatures):
  with open(path, "r") as f:
    data = json.load(f)
  return Player.from_dict(data, all_creatures)
