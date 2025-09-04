import numpy as np
import torch.optim as optim
from app.config import ACTION_NAMES, CONFIG, DOT_DAMAGE, SPECIAL_ABILITIES
from app.modules.logging_utils import append_battle_log
from app.modules.neural_network import NeuralNetwork

class Creature:
  def __init__(self, name, nn_model, config_stats):
    self.name = name
    self.nn = nn_model
    self.hp = config_stats['hp']
    self.max_hp = config_stats['max_hp']
    self.energy = config_stats['energy']
    self.max_energy = config_stats['max_energy']
    self.speed = config_stats['speed']
    self.special_abilities = config_stats['special_abilities']
    self.statuses = {}
    self.reward_config = config_stats.get('reward_config', {})

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

  def is_alive(self):
    return self.hp > 0

  def process_statuses(self, opponent, abl_zero_reward):
    """Apply DOT effects and call abl_zero_reward if needed."""
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


def init_creatures(creature_dict):
  creatures = {}
  optimizers = {}
  input_size = len(ACTION_NAMES)
  for name, stats in creature_dict.items():
    nn_output_size = 3 + len(stats['special_abilities'])
    nn = NeuralNetwork(input_size, CONFIG['hidden_sizes'], nn_output_size)
    creature = Creature(name, nn, stats)
    optimizer = optim.Adam(nn.parameters(), lr=CONFIG['learning_rate'])
    creatures[name] = creature
    optimizers[name] = optimizer
  return creatures, optimizers
