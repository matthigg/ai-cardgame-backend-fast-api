import os
import torch.optim as optim
from tqdm import trange
from app.config import ACTION_NAMES, CONFIG, CREATURES
from app.modules.creature import Creature
from app.modules.battle_simulation import simulate_battle
from app.modules.logging_utils import write_logs
from app.modules.neural_network import NeuralNetwork, reinforce_update
from app.modules.network_persistence import resume_from_checkpoint, save_best_checkpoint

# ------------------ Training Loop ------------------

def training_loop():
  os.makedirs(CONFIG['log_dir'], exist_ok=True)
  os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

  input_size = 4
  output_size = len(ACTION_NAMES)
  nn_A = NeuralNetwork(input_size, CONFIG['hidden_sizes'], output_size)
  nn_B = NeuralNetwork(input_size, CONFIG['hidden_sizes'], output_size)

  creature_A = Creature('A', nn_A, CREATURES['A'])
  creature_B = Creature('B', nn_B, CREATURES['B'])

  optimizer_A = optim.Adam(creature_A.nn.parameters(), lr=CONFIG['learning_rate'])
  optimizer_B = optim.Adam(creature_B.nn.parameters(), lr=CONFIG['learning_rate'])

  start_epoch = resume_from_checkpoint(creature_A, creature_B, optimizer_A, optimizer_B)
  baseline_A, baseline_B = 0.0, 0.0
  batched_logs = []
  batched_logs_total = []

  epsilon = CONFIG['epsilon']
  wins = {creature_A.name: 0, creature_B.name: 0}

  best_reward_A, best_reward_B = -float('inf'), -float('inf')

  # Track the last cumulative epoch reached
  last_epoch = start_epoch - 1

  # Each batch runs CONFIG['epochs'] times
  for batch_epoch in range(CONFIG['epochs']):
    epoch = start_epoch + batch_epoch
    last_epoch = epoch

    epsilon = max(CONFIG['eps_min'], epsilon * CONFIG['eps_decay_rate'])

    reward_A, reward_B, battle_log, winner = simulate_battle(
      creature_A, creature_B, epoch, CONFIG['batch_size'], epsilon
    )

    reinforce_update(creature_A, optimizer_A, battle_log, baseline_A, CONFIG['entropy_beta'])
    reinforce_update(creature_B, optimizer_B, battle_log, baseline_B, CONFIG['entropy_beta'])

    baseline_A = (1 - CONFIG['alpha_baseline']) * baseline_A + CONFIG['alpha_baseline'] * reward_A
    baseline_B = (1 - CONFIG['alpha_baseline']) * baseline_B + CONFIG['alpha_baseline'] * reward_B

    if winner:
      wins[winner] += 1
      best_reward_A, best_reward_B = save_best_checkpoint(
        epoch, best_reward_A, best_reward_B, reward_A, reward_B,
        creature_A, creature_B, optimizer_A, optimizer_B
      )

    batched_logs.append((epoch, battle_log, reward_A, reward_B, wins[creature_A.name], wins[creature_B.name]))
    batched_logs_total.append((epoch, battle_log, reward_A, reward_B, wins[creature_A.name], wins[creature_B.name]))

    if len(batched_logs) % CONFIG['batch_size'] == 0:
      write_logs(batched_logs, last_epoch, finalLog=False)
      batched_logs = []

  # Write final summary for this batch
  write_logs(batched_logs_total, last_epoch, finalLog=True, finalWins=wins)
