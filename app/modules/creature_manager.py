# app/modules/creature_manager.py
import os
import json
import itertools
import copy
import numpy as np
import torch
import torch.optim as optim
from app.config import ACTION_NAMES, CONFIG, CREATURE_TEMPLATES, DOT_DAMAGE, SPECIAL_ABILITIES
from app.modules.neural_network import NeuralNetwork

CREATURE_DIR = "creatures"
os.makedirs(CREATURE_DIR, exist_ok=True)

# Active creature registry (for creatures currently in memory)
# Key: "id_name"
_active_creatures: dict[str, 'Creature'] = {}

# global unique ID counter
_creature_id_counter = itertools.count(1)

def _make_key(creature_id: int, name: str) -> str:
  return f"{creature_id}_{name}"

class Creature:
  def __init__(self, name, owner, nn_model, config_stats, creature_id=None):
    self.id = creature_id or next(_creature_id_counter)
    self.name = name
    self.owner = owner
    self.nn = nn_model

    # Base stats
    self.hp = config_stats['hp']
    self.max_hp = config_stats['max_hp']
    self.energy = config_stats['energy']
    self.max_energy = config_stats['max_energy']
    self.speed = config_stats['speed']
    self.special_abilities = config_stats.get('special_abilities', [])
    self.reward_config = config_stats.get('reward_config', {})

    # Track NN activations for visualization
    self.activations_history: list[dict] = []

    # Runtime state
    self.statuses: dict[str, int] = {}
    self.runtime_state = {
      "hp": self.hp,
      "energy": self.energy,
      "statuses": self.statuses
    }

    # NN config
    self.nn_config = config_stats.get('nn_config', {})

    # Actions list
    self.actions = [
      ('attack', self.attack),
      ('defend', self.defend),
      ('recover', self.recover),
    ]
    for ability_name in self.special_abilities:
      self.actions.append((ability_name, self.use_special))


  def reset(self):
    self.hp = self.max_hp
    self.energy = self.max_energy
    self.statuses = {}
    self.runtime_state = {
      "hp": self.hp,
      "energy": self.energy,
      "statuses": self.statuses
    }

  def is_alive(self):
    return self.hp > 0

  def process_statuses(self, opponent, abl_zero_reward):
    for status in list(self.statuses.keys()):
      if status == 'poison':
        self.hp -= DOT_DAMAGE['poison_damage']
        if self.hp <= 0:
          abl_zero_reward(self, opponent, '*POISONED*', 99)
      self.statuses[status] -= 1
      if self.statuses[status] <= 0:
        del self.statuses[status]

  def attack(self, opponent):
    dmg = CONFIG['attack_damage']
    if 'defend' in opponent.statuses:
      dmg = int(np.ceil(dmg / 2))
    opponent.hp -= dmg
    self.energy = min(self.max_energy, self.energy + CONFIG['energy_regen_base'])
    return self.reward_config.get('attack', CONFIG['reward_attack'])

  def defend(self, opponent=None):
    self.statuses['defend'] = 1
    self.energy = min(self.max_energy, self.energy + CONFIG['energy_regen_base'])
    return self.reward_config.get('defend', CONFIG['reward_defend'])

  def use_special(self, opponent, ability_name):
    ability = SPECIAL_ABILITIES.get(ability_name)
    if ability and self.energy >= ability['energy_cost']:
      self.energy -= ability['energy_cost']
      ability['apply'](self, opponent)
      return self.reward_config.get(ability_name, 0.01)
    return 0.0

  def recover(self, opponent=None):
    if self.energy >= self.max_energy:
      return -self.reward_config.get('recover', CONFIG['reward_recover'])
    self.energy = min(self.max_energy, self.energy + CONFIG['energy_regen_recover'])
    return self.reward_config.get('recover', CONFIG['reward_recover'])

  def to_dict(self):
    return {
      "id": self.id,
      "name": self.name,
      "owner": self.owner,
      "hp": self.hp,
      "max_hp": self.max_hp,
      "energy": self.energy,
      "max_energy": self.max_energy,
      "speed": self.speed,
      "special_abilities": self.special_abilities,
      "reward_config": self.reward_config,
      "nn_config": self.nn_config,
      "runtime_state": self.runtime_state
    }

  @classmethod
  def from_dict(cls, data, nn_model):
    creature = cls(
      name=data['name'],
      owner=data['owner'],
      nn_model=nn_model,
      config_stats=data,
      creature_id=data['id']
    )
    return creature

def init_creatures(creature_dict):
  creatures = {}
  optimizers = {}
  input_size = len(ACTION_NAMES)

  for name, stats in creature_dict.items():
    nn_output_size = 3 + len(stats.get('special_abilities', []))
    nn = NeuralNetwork(input_size, stats.get('nn_config', {}).get('hidden_sizes', CONFIG['hidden_sizes']), nn_output_size)
    optimizer = optim.Adam(nn.parameters(), lr=stats.get('nn_config', {}).get('learning_rate', CONFIG['learning_rate']))
    creature = Creature(name, owner="SYSTEM", nn_model=nn, config_stats=stats)
    creatures[name] = creature
    optimizers[name] = optimizer
  return creatures, optimizers

def save_creature(creature: Creature):
  os.makedirs(CREATURE_DIR, exist_ok=True)
  path = os.path.join(CREATURE_DIR, f"creature_{creature.id}_{creature.name}.json")
  with open(path, "w") as f:
    json.dump(creature.to_dict(), f, indent=2)
  _active_creatures[_make_key(creature.id, creature.name)] = creature
  return path

def load_creature(creature_id, name, nn_model):
  path = os.path.join(CREATURE_DIR, f"creature_{creature_id}_{name}.json")
  if not os.path.exists(path):
    return None
  with open(path, "r") as f:
    data = json.load(f)
  creature = Creature.from_dict(data, nn_model)
  _active_creatures[_make_key(creature.id, creature.name)] = creature
  return creature

def add_active_creature(creature: Creature):
  key = _make_key(creature.id, creature.name)
  _active_creatures[key] = creature
  return creature

def remove_active_creature(creature_id, name):
  key = _make_key(creature_id, name)
  _active_creatures.pop(key, None)

def list_active_creatures():
  return list(_active_creatures.keys())

def create_creature(template_key, owner):
  template = CREATURE_TEMPLATES[template_key]
  nn_output_size = 3 + len(template.get('special_abilities', []))
  nn_model = NeuralNetwork(len(ACTION_NAMES), template.get('nn_config', {}).get('hidden_sizes', CONFIG['hidden_sizes']), nn_output_size)
  creature_id = next(_creature_id_counter)
  creature = Creature(template['name'], owner, nn_model, template, creature_id=creature_id)
  save_creature(creature)
  return creature
