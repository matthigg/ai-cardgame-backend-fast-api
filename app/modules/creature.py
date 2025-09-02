import numpy as np
import torch
import torch.nn.functional as F
from app.config import ACTION_NAMES, CONFIG, DOT_DAMAGE, SPECIAL_ABILITIES
from app.modules.logging_utils import append_battle_log

# ------------------ Creature ------------------

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
    self.action_rewards = 0.0

    # Available actions: (name, function)
    self.actions = [
      ('Attack', self.attack),
      ('Defend', self.defend),
      ('Special', self.use_special),
      ('Recover', self.recover)
    ]

  def is_alive(self):
    return self.hp > 0
  
  def process_statuses(self, epoch, tick, creature, opponent, battle_log, zero):
    for status in list(self.statuses.keys()):
      if status == 'Poison':
        self.hp -= DOT_DAMAGE['poison_damage']
        if self.hp <= 0:
          append_battle_log(epoch, tick, creature, opponent, battle_log, '*POISONED*', zero, -1, 0.0)

      # Handle status decay
      self.statuses[status] -= 1
      if self.statuses[status] <= 0:
        del self.statuses[status]

  def attack(self, opponent):
    dmg = CONFIG['attack_damage']
    if 'Defend' in opponent.statuses:
      dmg = int(np.ceil(dmg / 2))
    opponent.hp -= dmg
    self.energy = min(self.max_energy, self.energy + CONFIG['energy_regen_base'])
    return CONFIG['reward_attack']

  def defend(self, opponent=None):
    self.statuses['Defend'] = 1
    self.energy = min(self.max_energy, self.energy + CONFIG['energy_regen_base'])
    return CONFIG['reward_defend']

  def use_special(self, opponent):
    if len(self.special_abilities) == 0:
      return 0.0
    
    ability = SPECIAL_ABILITIES.get(self.special_abilities[0])

    if ability and self.energy >= ability['energy_cost']:
      self.energy -= ability['energy_cost']
      ability['apply'](self, opponent)
      return ability['reward']
    return 0.0

  def recover(self, opponent=None):
    if self.energy >= self.max_energy:
      return -CONFIG['reward_recover']
    self.energy = min(self.max_energy, self.energy + CONFIG['energy_regen_recover'])
    return CONFIG['reward_recover']