# app/modules/creature_manager.py
import os
import json
import torch
import numpy as np
from app.config import ACTION_NAMES, CONFIG, CREATURE_BASE_STATS, CREATURE_REWARD_CONFIG, CREATURE_TEMPLATES, DOT_DAMAGE, SPECIAL_ABILITIES
from app.modules.neural_network import NeuralNetwork
from app.modules.utils import get_player_json_path, get_checkpoint_path

class Creature:
  def __init__(self, name, owner, nn_model, config_stats, creature_id):
    self.id = creature_id
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

    # Runtime state
    self.statuses = {}
    self.runtime_state = {
      "hp": self.hp,
      "energy": self.energy,
      "statuses": self.statuses
    }

    # NN config
    self.nn_config = config_stats.get('nn_config', {})

    # Activations history
    self.activations_history = []

    # Actions list
    self.actions = [
      ('attack', self.attack),
      ('defend', self.defend),
      ('recover', self.recover),
    ]
    for ability_name in self.special_abilities:
      self.actions.append((ability_name, self.use_special))

  def attack(self, opponent):
    dmg = CREATURE_BASE_STATS['attack_damage']
    if 'defend' in opponent.statuses:
      dmg = int(np.ceil(dmg / 2))
    opponent.hp -= dmg
    self.energy = min(self.max_energy, self.energy + CREATURE_BASE_STATS['energy_regen_base'])
    return self.reward_config.get('attack', CREATURE_REWARD_CONFIG['attack'])

  def defend(self, opponent=None):
    self.statuses['defend'] = 1
    self.energy = min(self.max_energy, self.energy + CREATURE_BASE_STATS['energy_regen_base'])
    return self.reward_config.get('defend', CREATURE_REWARD_CONFIG['defend'])

  def use_special(self, opponent, ability_name):
    ability = SPECIAL_ABILITIES.get(ability_name)
    if ability and self.energy >= ability['energy_cost']:
      self.energy -= ability['energy_cost']
      ability['apply'](self, opponent)
      return self.reward_config.get(ability_name, 0.01)
    return 0.0

  def recover(self, opponent=None):
    if self.energy >= self.max_energy:
      return -self.reward_config.get('recover', CREATURE_REWARD_CONFIG['recover'])
    self.energy = min(self.max_energy, self.energy + CREATURE_BASE_STATS['energy_regen_recover'])
    return self.reward_config.get('recover', CREATURE_REWARD_CONFIG['recover'])

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
  
  def reset(self):
    self.hp = self.max_hp
    self.energy = self.max_energy
    self.statuses = {}
    self.runtime_state = {
      "hp": self.hp,
      "energy": self.energy,
      "statuses": self.statuses
    }

  def to_dict(self):
    """Serialize creature for player.json (checkpoint path stored separately)."""
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
    return cls(
      name=data['name'],
      owner=data['owner'],
      nn_model=nn_model,
      config_stats=data,
      creature_id=data['id']
    )
  
# ------------------ Creature neural network functions ------------------

def build_nn_for_creature(config_stats):
  input_size = len(ACTION_NAMES)
  output_size = 3 + len(config_stats.get('special_abilities', []))
  hidden_sizes = config_stats.get('nn_config', {}).get('hidden_sizes')
  nn_model = NeuralNetwork(input_size, hidden_sizes, output_size)
  return nn_model

# ------------------ Creature fetch functions ------------------

def fetch_creature_from_template(template_key: str, owner: str, creature_id: int):
  """Build a Creature from a template (used when player.json or creature not found)."""
  template = CREATURE_TEMPLATES[template_key]
  nn_model = build_nn_for_creature(template)
  creature = Creature(
    name=template['name'],
    owner=owner,
    nn_model=nn_model,
    config_stats=template,
    creature_id=creature_id
  )
  return creature

def fetch_creature_from_player_json(
  player_name: str,
  player_id: int,
  creature_id: int
):

  """Load a Creature from player.json and resume from checkpoint."""
  player_path = get_player_json_path(player_name, player_id)
  if not os.path.exists(player_path):
    return None

  with open(player_path, "r") as f:
    player_data = json.load(f)

  creature_entry = next((c for c in player_data['creatures'] if c['id'] == creature_id), None)
  if not creature_entry:
    return None

  # Build NN and load checkpoint
  nn_model = build_nn_for_creature(creature_entry)
  checkpoint_path = creature_entry.get('nn_checkpoint')
  if checkpoint_path and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    nn_model.load_state_dict(checkpoint['model_state_dict'])
  return Creature.from_dict(creature_entry, nn_model)
