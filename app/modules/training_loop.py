# app/modules/training_loop.py
import os
import torch
from app.config import CONFIG
from app.modules.creature_manager import Creature, fetch_creature_from_player_json
from app.modules.battle_simulation import simulate_battle
from app.modules.logging_utils import write_logs
from app.modules.neural_network import reinforce_update
from app.modules.network_persistence import resume_from_checkpoint, save_checkpoints
from app.modules.utils import get_checkpoint_path

def capture_activations(creature, input_tensor):
  """Return normalized neuron activations for visualization."""
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
  player_name_A: str, player_id_A: int, creature_name_A: str, creature_id_A: int,
  player_name_B: str, player_id_B: int, creature_name_B: str, creature_id_B: int,
):

  """Run training loop between two creatures defined in player JSON files."""

  creature_A = fetch_creature_from_player_json(player_name_A, player_id_A, creature_id_A)
  creature_B = fetch_creature_from_player_json(player_name_B, player_id_B, creature_id_B)

  if creature_A is None:
    raise ValueError(f"Creature {creature_id_A} for player {player_name_A} not found.")
  if creature_B is None:
    raise ValueError(f"Creature {creature_id_B} for player {player_name_B} not found.")

  optimizer_A = torch.optim.Adam(
    creature_A.nn.parameters(),
    lr=creature_A.nn_config.get('learning_rate', 0.001)
  )
  optimizer_B = torch.optim.Adam(
    creature_B.nn.parameters(),
    lr=creature_B.nn_config.get('learning_rate', 0.001)
  )

  print('============= RESUME ==========================')
  # Resume from checkpoint
  resume_from_checkpoint(
    creature_A, creature_B, optimizer_A, optimizer_B,
    player_name_A, player_id_A, player_name_B, player_id_B
  )

  nn_config_A = creature_A.nn_config
  nn_config_B = creature_B.nn_config

  baseline_A, baseline_B = 0.0, 0.0
  epsilon_A = nn_config_A.get('epsilon', 0.9)
  epsilon_B = nn_config_B.get('epsilon', 0.9)
  wins = {creature_A.name: 0, creature_B.name: 0}

  batched_logs, batched_logs_total = [], []

  for epoch in range(CONFIG['epoch_batch_size']):
    epsilon_A = max(nn_config_A.get('eps_min', 0.05), epsilon_A * nn_config_A.get('eps_decay_rate', 0.99))
    epsilon_B = max(nn_config_B.get('eps_min', 0.05), epsilon_B * nn_config_B.get('eps_decay_rate', 0.99))

    reward_A, reward_B, battle_log, winner, state_tensor_A, state_tensor_B = simulate_battle(
      creature_A, creature_B, epoch, CONFIG['max_ticks'], (epsilon_A, epsilon_B)
    )

    if winner and winner != 'stalemate':
      if winner in wins:
        wins[winner] += 1

    reinforce_update(creature_A, optimizer_A, battle_log, baseline_A, nn_config_A.get('entropy_beta', 0.0))
    reinforce_update(creature_B, optimizer_B, battle_log, baseline_B, nn_config_B.get('entropy_beta', 0.0))

    alpha_A = nn_config_A.get('alpha_baseline', 0.0)
    alpha_B = nn_config_B.get('alpha_baseline', 0.0)

    baseline_A = (1 - alpha_A) * baseline_A + alpha_A * reward_A
    baseline_B = (1 - alpha_B) * baseline_B + alpha_B * reward_B

    batched_logs.append((epoch, battle_log, reward_A, reward_B,
                         wins.get(creature_A.name, 0), wins.get(creature_B.name, 0)))
    batched_logs_total.append((epoch, battle_log, reward_A, reward_B,
                               wins.get(creature_A.name, 0), wins.get(creature_B.name, 0)))

    if state_tensor_A is not None:
      creature_A.activations_history.append({
        "name": creature_A.name,
        "epoch": epoch,
        "layers": capture_activations(creature_A, state_tensor_A)
      })
    if state_tensor_B is not None:
      creature_B.activations_history.append({
        "name": creature_B.name,
        "epoch": epoch,
        "layers": capture_activations(creature_B, state_tensor_B)
      })

    if len(batched_logs) % CONFIG['max_ticks'] == 0:
      write_logs(batched_logs, {}, finalLog=False)
      batched_logs = []

  print('============= SAVE ==========================')
  # Save checkpoints including optimizer and activations
  save_checkpoints(
    creature_A, creature_B, optimizer_A, optimizer_B,
    player_name_A, player_id_A, player_name_B, player_id_B
  )

  last_epochs = {
    creature_A.name: getattr(creature_A, 'current_epoch', 0),
    creature_B.name: getattr(creature_B, 'current_epoch', 0)
  }

  summary_data = write_logs(batched_logs_total, last_epochs, finalLog=True, final_wins=wins)

  return {
    "summary": summary_data,
    "activations": {
      creature_A.name: creature_A.activations_history,
      creature_B.name: creature_B.activations_history
    }
  }
