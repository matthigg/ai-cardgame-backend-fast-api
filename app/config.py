import torch
import numpy as np

# ------------------ Configuration ------------------

ACTION_NAMES = ['attack', 'defend', 'special', 'recover']

CONFIG = {
  'seed': 43,
  'use_seed': True,
  'epoch_batch_size': 100,
  'max_ticks': 50,

  'resume_from_checkpoint': True,
  'checkpoint_dir': 'checkpoints',
  'log_dir': 'battle_logs',

  'write_battle_logs': True,
  'write_battle_summary_log': True,
  'sort_logs_by_creature': False,

  'attack_damage': 20,
  'energy_regen_base': 10,
  'energy_regen_recover': 60,
}

DOT_DAMAGE = {
  'poison_damage': 5
}

SPECIAL_ABILITIES = {
  'poison': {
    'energy_cost': 30,
    'apply': lambda c, o: o.statuses.update({'poison': 3})
  },
  'stun': {
    'energy_cost': 40,
    'apply': lambda c, o: o.statuses.update({'stun': 2}) if 'defend' not in o.statuses else None
  }
}

CREATURE_NN_CONFIG = {
  'hidden_sizes': [4, 4, 20, 3],
  'learning_rate': 0.001,
  'epsilon': 0.9,
  'eps_min': 0.05,
  'eps_decay_rate': 0.99,
  'alpha_baseline': 0.05,
  'entropy_beta': 0.001,
  'max_display_neurons': 5
}

CREATURE_REWARD_CONFIG = {
  'attack': 0.01,
  'defend': 0.01,
  'recover': 0.01,
  'win': 10,
  'lose': -10,
  'poison': 0.01,
  'stun': 0.01
}

CREATURE_TEMPLATES = {
  'A': {
    'name': 'A',
    'creature_template_id': 1,
    'hp': 100,
    'max_hp': 100,
    'energy': 100,
    'max_energy': 100,
    'speed': 10,
    'special_abilities': ['stun'],
    'nn_config': CREATURE_NN_CONFIG,
    'reward_config': CREATURE_REWARD_CONFIG,
  },
  'B': {
    'name': 'B',
    'creature_template_id': 2,
    'hp': 100,
    'max_hp': 100,
    'energy': 100,
    'max_energy': 100,
    'speed': 10,
    'special_abilities': ['poison'],
    'nn_config': CREATURE_NN_CONFIG,
    'reward_config': CREATURE_REWARD_CONFIG,
  }
}

PLAYER_TEMPLATES = {
  1: {  # Player ID as the key
    "id": 1,
    "name": "Alice",
    "creatures": [
      { "template": "A", "id": 101 }  # Creature ID
    ]
  },
  2: {
    "id": 2,
    "name": "Bob",
    "creatures": [
      { "template": "B", "id": 102 }
    ]
  }
}



if CONFIG['use_seed']:
  np.random.seed(CONFIG['seed'])
  torch.manual_seed(CONFIG['seed'])
