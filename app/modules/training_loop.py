# app/modules/training_loop.py
import os
import torch
from app.config import BATTLE_LOGS_DIR, CONFIG
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

def _parse_player_path(player_path: str) -> tuple[str, int]:
  """Extract (player_name, player_id) from a player JSON file path."""
  basename = os.path.splitext(os.path.basename(player_path))[0]
  if "_" not in basename:
    raise ValueError(f"Invalid player file name format: {basename}")
  name, id_str = basename.rsplit("_", 1)
  try:
    pid = int(id_str)
  except ValueError:
    raise ValueError(f"Invalid player id in filename: {id_str}")
  return name, pid

def training_loop(playerA_path: str, playerB_path: str, creature_id_A: int, creature_id_B: int):
  """Run training loop between two creatures defined in player JSON files."""
  os.makedirs(BATTLE_LOGS_DIR, exist_ok=True)

  playerA_name, playerA_id = _parse_player_path(playerA_path)
  playerB_name, playerB_id = _parse_player_path(playerB_path)

  creature_A = fetch_creature_from_player_json(playerA_name, playerA_id, creature_id_A)
  creature_B = fetch_creature_from_player_json(playerB_name, playerB_id, creature_id_B)

  if creature_A is None:
    raise ValueError(f"Creature {creature_id_A} for player {playerA_name} not found.")
  if creature_B is None:
    raise ValueError(f"Creature {creature_id_B} for player {playerB_name} not found.")

  optimizer_A = torch.optim.Adam(
    creature_A.nn.parameters(),
    lr=creature_A.nn_config.get('learning_rate', 0.001)
  )
  optimizer_B = torch.optim.Adam(
    creature_B.nn.parameters(),
    lr=creature_B.nn_config.get('learning_rate', 0.001)
  )

  resume_from_checkpoint(
    creature_A, creature_B, optimizer_A, optimizer_B,
    playerA_name, playerA_id, playerB_name, playerB_id
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

  save_checkpoints(
    creature_A, creature_B, optimizer_A, optimizer_B,
    playerA_name, playerA_id, playerB_name, playerB_id
  )

  A_path = get_checkpoint_path(playerA_name, playerA_id, creature_A.name, creature_A.id)
  B_path = get_checkpoint_path(playerB_name, playerB_id, creature_B.name, creature_B.id)

  if os.path.exists(A_path):
    checkpoint_A = torch.load(A_path)
    checkpoint_A['activations_history'] = creature_A.activations_history
    torch.save(checkpoint_A, A_path)
  if os.path.exists(B_path):
    checkpoint_B = torch.load(B_path)
    checkpoint_B['activations_history'] = creature_B.activations_history
    torch.save(checkpoint_B, B_path)

  last_epochs = {
    creature_A.name: (torch.load(A_path).get('epoch', 0) if os.path.exists(A_path) else 0),
    creature_B.name: (torch.load(B_path).get('epoch', 0) if os.path.exists(B_path) else 0)
  }

  summary_data = write_logs(batched_logs_total, last_epochs, finalLog=True, final_wins=wins)

  return {
    "summary": summary_data,
    "activations": {
      creature_A.name: creature_A.activations_history,
      creature_B.name: creature_B.activations_history
    }
  }
