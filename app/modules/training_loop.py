import os
import copy
import torch
from app.config import CONFIG, PLAYER_TEMPLATES, CREATURE_TEMPLATES
from app.modules.creature_manager import Creature, save_creature, create_creature
from app.modules.battle_simulation import simulate_battle
from app.modules.logging_utils import write_logs
from app.modules.neural_network import reinforce_update, NeuralNetwork
from app.modules.network_persistence import resume_from_checkpoint, save_checkpoints
from app.modules.utils import create_checkpoint_path

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

def training_loop():
  os.makedirs(CONFIG['log_dir'], exist_ok=True)
  os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

  # --- Hardcode Alice and Bob's creatures ---
  alice_creature_data = PLAYER_TEMPLATES[1]['creatures'][0]
  bob_creature_data   = PLAYER_TEMPLATES[2]['creatures'][0]

  # Create creature instances using helper
  creature_A = create_creature('A', 'Alice', creature_id=alice_creature_data['id'])
  creature_B = create_creature('B', 'Bob', creature_id=bob_creature_data['id'])

  # Get creature templates
  template_A = CREATURE_TEMPLATES['A']
  template_B = CREATURE_TEMPLATES['B']

  save_creature(creature_A)
  save_creature(creature_B)

  optimizer_A = torch.optim.Adam(
    creature_A.nn.parameters(),
    lr=template_A['nn_config']['learning_rate']
  )
  optimizer_B = torch.optim.Adam(
    creature_B.nn.parameters(),
    lr=template_B['nn_config']['learning_rate']
  )

  # Resume from checkpoints
  resume_from_checkpoint(creature_A, creature_B, optimizer_A, optimizer_B)

  nn_config_A = creature_A.nn_config
  nn_config_B = creature_B.nn_config

  baseline_A, baseline_B = 0.0, 0.0
  epsilon_A = nn_config_A['epsilon']
  epsilon_B = nn_config_B['epsilon']
  wins = {creature_A.name: 0, creature_B.name: 0}

  batched_logs, batched_logs_total = [], []

  for epoch in range(CONFIG['epoch_batch_size']):
    epsilon_A = max(nn_config_A['eps_min'], epsilon_A * nn_config_A['eps_decay_rate'])
    epsilon_B = max(nn_config_B['eps_min'], epsilon_B * nn_config_B['eps_decay_rate'])

    reward_A, reward_B, battle_log, winner, state_tensor_A, state_tensor_B = simulate_battle(
      creature_A, creature_B, epoch, CONFIG['max_ticks'], (epsilon_A, epsilon_B)
    )

    if winner and winner != 'stalemate':
      wins[winner] += 1

    reinforce_update(
      creature_A, optimizer_A, battle_log, baseline_A, nn_config_A['entropy_beta']
    )
    reinforce_update(
      creature_B, optimizer_B, battle_log, baseline_B, nn_config_B['entropy_beta']
    )

    alpha_A = nn_config_A['alpha_baseline']
    alpha_B = nn_config_B['alpha_baseline']

    baseline_A = (1 - alpha_A) * baseline_A + alpha_A * reward_A
    baseline_B = (1 - alpha_B) * baseline_B + alpha_B * reward_B

    batched_logs.append((epoch, battle_log, reward_A, reward_B,
                         wins[creature_A.name], wins[creature_B.name]))
    batched_logs_total.append((epoch, battle_log, reward_A, reward_B,
                               wins[creature_A.name], wins[creature_B.name]))

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

  # Save training-specific checkpoints
  save_checkpoints(creature_A, creature_B, optimizer_A, optimizer_B)
  A_path = create_checkpoint_path(creature_A)
  B_path = create_checkpoint_path(creature_B)

  checkpoint_A = torch.load(A_path)
  checkpoint_A['activations_history'] = creature_A.activations_history
  torch.save(checkpoint_A, A_path)

  checkpoint_B = torch.load(B_path)
  checkpoint_B['activations_history'] = creature_B.activations_history
  torch.save(checkpoint_B, B_path)

  last_epochs = {
    creature_A.name: checkpoint_A.get('epoch', 0),
    creature_B.name: checkpoint_B.get('epoch', 0)
  }
  summary_data = write_logs(batched_logs_total, last_epochs, finalLog=True, final_wins=wins)

  return {
    "summary": summary_data,
    "activations": {
      creature_A.name: creature_A.activations_history,
      creature_B.name: creature_B.activations_history
    }
  }
