import torch
import numpy as np

# ------------------ Configuration ------------------

ACTION_NAMES = ['attack', 'defend', 'special', 'recover']

CHECKPOINT_DIR = "checkpoints"
GENERATED_DIR = 'generated'
BATTLE_LOGS_DIR = 'battle_logs'
PLAYERS_DIR = "players"

CONFIG = {
  'seed': 43,
  'use_seed': True,
  'epoch_batch_size': 100,
  'max_ticks': 50,

  'resume_from_checkpoint': True,
  'log_dir': 'battle_logs',

  'write_battle_logs': True,
  'write_battle_summary_log': True,
  'sort_logs_by_creature': False,
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

CREATURE_BASE_STATS = {
  'attack_damage': 20,
  'energy_regen_base': 10,
  'energy_regen_recover': 60,
  'hp': 100,
  'max_hp': 100,
  'energy': 100,
  'max_energy': 100,
  'speed': 10,
}

CREATURE_NN_CONFIG = {
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
  'Bear': {
    'name': 'Bear',
    'creature_template_id': 1,
    **CREATURE_BASE_STATS,
    'special_abilities': ['stun'],
    'nn_config': {
      **CREATURE_NN_CONFIG,
    'hidden_sizes': [4, 8, 10, 3],
    },
    'reward_config': CREATURE_REWARD_CONFIG,
  },
  'Snake': {
    'name': 'Snake',
    'creature_template_id': 2,
    **CREATURE_BASE_STATS,
    'special_abilities': ['poison'],
    'nn_config': {
      **CREATURE_NN_CONFIG,
      'hidden_sizes': [8, 6, 4],
    },
    'reward_config': CREATURE_REWARD_CONFIG,
  }
}




if CONFIG['use_seed']:
  np.random.seed(CONFIG['seed'])
  torch.manual_seed(CONFIG['seed'])
