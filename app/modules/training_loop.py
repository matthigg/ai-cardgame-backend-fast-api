import os
import torch.optim as optim
from tqdm import trange
from app.config import ACTION_NAMES, CONFIG, CREATURES
from app.modules.creature import Creature, init_creatures
from app.modules.battle_simulation import simulate_battle
from app.modules.logging_utils import write_logs
from app.modules.neural_network import NeuralNetwork, reinforce_update
from app.modules.network_persistence import resume_from_checkpoint, save_checkpoints

# ------------------ Training Loop ------------------

def training_loop():
  os.makedirs(CONFIG['log_dir'], exist_ok=True)
  os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

  creatures, optimizers = init_creatures(CREATURES)
  creature_names = list(creatures.keys())[:2]  # pick first 2 creatures
  creature_A, creature_B = creatures[creature_names[0]], creatures[creature_names[1]]
  optimizer_A, optimizer_B = optimizers[creature_names[0]], optimizers[creature_names[1]]

  resume_from_checkpoint(creature_A, creature_B, optimizer_A, optimizer_B)

  baseline_A, baseline_B = 0.0, 0.0
  batched_logs, batched_logs_total = [], []

  epsilon = CONFIG['epsilon']
  wins = {creature_A.name: 0, creature_B.name: 0}

  for epoch in range(CONFIG['epoch_batch_size']):
    epsilon = max(CONFIG['eps_min'], epsilon * CONFIG['eps_decay_rate'])

    reward_A, reward_B, battle_log, winner = simulate_battle(
      creature_A, creature_B, epoch, CONFIG['log_batch_size'], epsilon
    )

    if winner:
      wins[winner] += 1

    reinforce_update(creature_A, optimizer_A, battle_log, baseline_A, CONFIG['entropy_beta'])
    reinforce_update(creature_B, optimizer_B, battle_log, baseline_B, CONFIG['entropy_beta'])

    baseline_A = (1 - CONFIG['alpha_baseline']) * baseline_A + CONFIG['alpha_baseline'] * reward_A
    baseline_B = (1 - CONFIG['alpha_baseline']) * baseline_B + CONFIG['alpha_baseline'] * reward_B

    batched_logs.append((epoch, battle_log, reward_A, reward_B, wins[creature_A.name], wins[creature_B.name]))
    batched_logs_total.append((epoch, battle_log, reward_A, reward_B, wins[creature_A.name], wins[creature_B.name]))

    if len(batched_logs) % CONFIG['log_batch_size'] == 0:
      write_logs(batched_logs, epoch, finalLog=False)
      batched_logs = []

  save_checkpoints(creature_A, creature_B, optimizer_A, optimizer_B)
  write_logs(batched_logs_total, CONFIG['epoch_batch_size'], finalLog=True, finalWins=wins)
