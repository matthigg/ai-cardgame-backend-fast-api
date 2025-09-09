import os
import copy
import torch
from typing import Optional
from app.config import CONFIG, CREATURE_TEMPLATES
from app.modules.creature_manager import init_creatures, Creature
from app.modules.battle_simulation import simulate_battle
from app.modules.logging_utils import write_logs
from app.modules.neural_network import reinforce_update
from app.modules.network_persistence import resume_from_checkpoint, save_checkpoints
from app.modules.utils import create_checkpoint_paths

def capture_activations(creature: Creature, input_tensor: torch.Tensor):
  """Return a list of neuron activations (layer outputs) for visualization."""
  activations = []

  def forward_hook(module, input, output):
    if isinstance(output, torch.Tensor):
      flat = output.detach().cpu().flatten()
      if flat.numel() > 0:
        min_val = flat.min()
        max_val = flat.max()
        normalized = (flat - min_val) / (max_val - min_val + 1e-8)
        activations.append(normalized.tolist())
      else:
        activations.append(flat.tolist())

  hooks = []
  for module in creature.nn.modules():
    if isinstance(module, torch.nn.Linear):
      hooks.append(module.register_forward_hook(forward_hook))

  creature.nn(input_tensor)

  for hook in hooks:
    hook.remove()

  return activations

def training_loop(
  player_creature: Creature,
  opponent_creature: Optional[Creature] = None,
  epochs: int = CONFIG['epoch_batch_size']
):
  """Run training for a specific player's creature against an opponent (player or NPC)."""
  os.makedirs(CONFIG['log_dir'], exist_ok=True)
  os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

  # Clone the player's creature for training
  train_creature = copy.deepcopy(player_creature)
  train_creature.reset()
  train_creature.activations_history = []

  # Clone or create opponent creature
  if opponent_creature is None:
    base_creatures, optimizers = init_creatures(CREATURE_TEMPLATES)
    opponent_name = list(base_creatures.keys())[0]
    train_opponent = copy.deepcopy(base_creatures[opponent_name])
    optimizer_opponent = optimizers[opponent_name]
  else:
    train_opponent = copy.deepcopy(opponent_creature)
    optimizer_opponent = torch.optim.Adam(train_opponent.nn.parameters(), lr=CONFIG['learning_rate'])

  train_opponent.reset()
  train_opponent.activations_history = []

  # Setup optimizer for player's creature
  optimizer_train = torch.optim.Adam(train_creature.nn.parameters(), lr=CONFIG['learning_rate'])

  # Resume from checkpoints if available
  resume_from_checkpoint(train_creature, train_opponent, optimizer_train, optimizer_opponent)

  nn_train_conf = getattr(train_creature, 'nn_config', {})
  nn_opponent_conf = getattr(train_opponent, 'nn_config', {})

  baseline_train, baseline_opponent = 0.0, 0.0
  epsilon_train = nn_train_conf.get('epsilon', CONFIG['epsilon'])
  epsilon_opponent = nn_opponent_conf.get('epsilon', CONFIG['epsilon'])
  wins = {train_creature.name: 0, train_opponent.name: 0}

  batched_logs_total = []

  for epoch in range(epochs):
    epsilon_train = max(nn_train_conf.get('eps_min', CONFIG['eps_min']),
                        epsilon_train * nn_train_conf.get('eps_decay_rate', CONFIG['eps_decay_rate']))
    epsilon_opponent = max(nn_opponent_conf.get('eps_min', CONFIG['eps_min']),
                           epsilon_opponent * nn_opponent_conf.get('eps_decay_rate', CONFIG['eps_decay_rate']))

    reward_train, reward_opponent, battle_log, winner, state_tensor_train, state_tensor_opponent = simulate_battle(
      train_creature, train_opponent, epoch, CONFIG['max_ticks'], (epsilon_train, epsilon_opponent)
    )

    if winner and winner != 'stalemate':
      wins[winner] += 1

    # Reinforce updates
    reinforce_update(train_creature, optimizer_train, battle_log, baseline_train,
                     nn_train_conf.get('entropy_beta', CONFIG['entropy_beta']))
    reinforce_update(train_opponent, optimizer_opponent, battle_log, baseline_opponent,
                     nn_opponent_conf.get('entropy_beta', CONFIG['entropy_beta']))

    # Update baselines
    baseline_train = (1 - nn_train_conf.get('alpha_baseline', CONFIG['alpha_baseline'])) * baseline_train + \
                     nn_train_conf.get('alpha_baseline', CONFIG['alpha_baseline']) * reward_train
    baseline_opponent = (1 - nn_opponent_conf.get('alpha_baseline', CONFIG['alpha_baseline'])) * baseline_opponent + \
                        nn_opponent_conf.get('alpha_baseline', CONFIG['alpha_baseline']) * reward_opponent

    batched_logs_total.append((epoch, battle_log, reward_train, reward_opponent,
                               wins[train_creature.name], wins[train_opponent.name]))

    # Capture activations
    if state_tensor_train is not None:
      train_creature.activations_history.append({
        "name": train_creature.name,
        "epoch": epoch,
        "layers": capture_activations(train_creature, state_tensor_train)
      })
    if state_tensor_opponent is not None:
      train_opponent.activations_history.append({
        "name": train_opponent.name,
        "epoch": epoch,
        "layers": capture_activations(train_opponent, state_tensor_opponent)
      })

  # Save checkpoints
  save_checkpoints(train_creature, train_opponent, optimizer_train, optimizer_opponent)
  A_path, B_path = create_checkpoint_paths(train_creature, train_opponent)
  checkpoint_A = torch.load(A_path)
  checkpoint_A['activations_history'] = train_creature.activations_history
  torch.save(checkpoint_A, A_path)
  checkpoint_B = torch.load(B_path)
  checkpoint_B['activations_history'] = train_opponent.activations_history
  torch.save(checkpoint_B, B_path)

  last_epochs = {train_creature.name: checkpoint_A.get('epoch', 0),
                 train_opponent.name: checkpoint_B.get('epoch', 0)}
  summary_data = write_logs(batched_logs_total, last_epochs, finalLog=True, final_wins=wins)

  return {
    "summary": summary_data,
    "activations": {
      train_creature.name: train_creature.activations_history,
      train_opponent.name: train_opponent.activations_history
    }
  }
