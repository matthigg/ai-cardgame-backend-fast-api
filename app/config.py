import torch
import numpy as np

# ------------------ Configuration ------------------

ACTION_NAMES = ['attack', 'defend', 'special', 'recover']

CONFIG = {
  'seed': 43,
  'use_seed': True,
  'epoch_batch_size': 100,
  'log_batch_size': 50,

  'resume_from_checkpoint': True,
  'checkpoint_dir': 'checkpoints',
  'log_dir': 'battle_logs',

  'write_battle_logs': False,
  'write_battle_summary_log': True,

  'epsilon': 0.9,
  'eps_min': 0.05,
  'eps_decay_rate': 0.99,
  'alpha_baseline': 0.05,
  'entropy_beta': 0.001,
  'learning_rate': 0.001,
  'hidden_sizes': [64, 32, 16],

  'attack_damage': 20,
  'energy_regen_base': 10,
  'energy_regen_recover': 60,

  'reward_attack': 0.01,
  'reward_defend': 0.01,
  'reward_recover': 0.01,
  'reward_win': 10,
  'reward_lose': -10
}

DOT_DAMAGE = {
  'poison_damage': 5
}

SPECIAL_ABILITIES = {
  'poison': {
    'energy_cost': 30,
    'apply': lambda c, o: o.statuses.update({'poison': 3}), 
    'reward': 0.01
  },
  'stun': {
    'energy_cost': 40,
    'apply': lambda c, o: o.statuses.update({'stun': 2}) if 'defend' not in o.statuses else None,
    'reward': 0.01
  }
}

CREATURES = {
  'C': {
    'id': 1,
    'hp': 100,
    'max_hp': 100,
    'energy': 100,
    'max_energy': 100,
    'speed': 10,
    'special_abilities': ['stun', 'poison']
  },
  'B': {
    'id': 2,
    'hp': 100,
    'max_hp': 100,
    'energy': 100,
    'max_energy': 100,
    'speed': 10,
    'special_abilities': ['stun', 'poison']
  },
}

if CONFIG['use_seed']:
  np.random.seed(CONFIG['seed'])
  torch.manual_seed(CONFIG['seed'])








