# app/modules/creature_manager.py
import numpy as np
import torch
from app.config import ACTION_NAMES, CONFIG, DOT_DAMAGE, SPECIAL_ABILITIES
from app.modules.neural_network import NeuralNetwork

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

    # Activations history (for training visualization)
    self.activations_history = []

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
    """Serialize creature for storing in player.json (checkpoint path stored separately)."""
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
    """Instantiate Creature from player.json data + loaded NN."""
    return cls(
      name=data['name'],
      owner=data['owner'],
      nn_model=nn_model,
      config_stats=data,
      creature_id=data['id']
    )

def build_nn_for_creature(config_stats):
  """Helper to create a NeuralNetwork for a creature based on its config."""
  input_size = len(ACTION_NAMES)
  output_size = 3 + len(config_stats.get('special_abilities', []))
  hidden_sizes = config_stats.get('nn_config', {}).get('hidden_sizes', CONFIG['hidden_sizes'])
  nn_model = NeuralNetwork(input_size, hidden_sizes, output_size)
  return nn_model
