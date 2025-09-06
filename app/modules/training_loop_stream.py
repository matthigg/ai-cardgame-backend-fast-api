import os
import torch
from app.config import CONFIG, CREATURES
from app.modules.creature import init_creatures
from app.modules.battle_simulation import simulate_battle
from app.modules.logging_utils import write_logs
from app.modules.neural_network import reinforce_update
from app.modules.network_persistence import resume_from_checkpoint, save_checkpoints
from app.modules.utils import create_checkpoint_paths

def capture_activations(creature, input_tensor):
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
    # Hook only Linear layers to avoid duplicates from activations
    if isinstance(module, torch.nn.Linear):
      hooks.append(module.register_forward_hook(forward_hook))

  # Run the network on the provided input
  creature.nn(input_tensor)

  for hook in hooks:
    hook.remove()

  return activations

def training_loop():
  """Run full training loop and return summary + activations."""
  os.makedirs(CONFIG['log_dir'], exist_ok=True)
  os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

  creatures, optimizers = init_creatures(CREATURES)
  creature_names = list(creatures.keys())[:2]
  creature_A, creature_B = creatures[creature_names[0]], creatures[creature_names[1]]
  optimizer_A, optimizer_B = optimizers[creature_names[0]], optimizers[creature_names[1]]

  resume_from_checkpoint(creature_A, creature_B, optimizer_A, optimizer_B)

  nn_config_A = getattr(creature_A, 'nn_config', {})
  nn_config_B = getattr(creature_B, 'nn_config', {})

  baseline_A, baseline_B = 0.0, 0.0
  epsilon_A = nn_config_A.get('epsilon', CONFIG['epsilon'])
  epsilon_B = nn_config_B.get('epsilon', CONFIG['epsilon'])
  wins = {creature_A.name: 0, creature_B.name: 0}

  batched_logs, batched_logs_total = [], []

  creature_A.activations_history = []
  creature_B.activations_history = []

  for epoch in range(CONFIG['epoch_batch_size']):
    # Decay epsilons
    epsilon_A = max(nn_config_A.get('eps_min', CONFIG['eps_min']),
                    epsilon_A * nn_config_A.get('eps_decay_rate', CONFIG['eps_decay_rate']))
    epsilon_B = max(nn_config_B.get('eps_min', CONFIG['eps_min']),
                    epsilon_B * nn_config_B.get('eps_decay_rate', CONFIG['eps_decay_rate']))

    # Run battle and get input tensors for activations
    reward_A, reward_B, battle_log, winner, state_tensor_A, state_tensor_B = simulate_battle(
      creature_A, creature_B, epoch, CONFIG['max_ticks'], (epsilon_A, epsilon_B)
    )

    if winner and winner != 'stalemate':
      wins[winner] += 1

    # Reinforce updates
    reinforce_update(creature_A, optimizer_A, battle_log, baseline_A,
                     nn_config_A.get('entropy_beta', CONFIG['entropy_beta']))
    reinforce_update(creature_B, optimizer_B, battle_log, baseline_B,
                     nn_config_B.get('entropy_beta', CONFIG['entropy_beta']))

    # Update baselines
    baseline_A = (1 - nn_config_A.get('alpha_baseline', CONFIG['alpha_baseline'])) * baseline_A + \
                 nn_config_A.get('alpha_baseline', CONFIG['alpha_baseline']) * reward_A
    baseline_B = (1 - nn_config_B.get('alpha_baseline', CONFIG['alpha_baseline'])) * baseline_B + \
                 nn_config_B.get('alpha_baseline', CONFIG['alpha_baseline']) * reward_B

    # Store logs
    batched_logs.append((epoch, battle_log, reward_A, reward_B,
                         wins[creature_A.name], wins[creature_B.name]))
    batched_logs_total.append((epoch, battle_log, reward_A, reward_B,
                               wins[creature_A.name], wins[creature_B.name]))

    # Capture activations for visualization
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

    # Periodic log write
    if len(batched_logs) % CONFIG['max_ticks'] == 0:
      write_logs(batched_logs, {}, finalLog=False)
      batched_logs = []

  # Save checkpoints including activations_history
  save_checkpoints(creature_A, creature_B, optimizer_A, optimizer_B)
  A_path, B_path = create_checkpoint_paths(creature_A, creature_B)
  checkpoint_A = torch.load(A_path)
  checkpoint_A['activations_history'] = creature_A.activations_history
  torch.save(checkpoint_A, A_path)
  checkpoint_B = torch.load(B_path)
  checkpoint_B['activations_history'] = creature_B.activations_history
  torch.save(checkpoint_B, B_path)

  last_epochs = {creature_A.name: checkpoint_A.get('epoch', 0),
                 creature_B.name: checkpoint_B.get('epoch', 0)}
  summary_data = write_logs(batched_logs_total, last_epochs, finalLog=True, final_wins=wins)

  return {
    "summary": summary_data,
    "activations": {
      creature_A.name: creature_A.activations_history,
      creature_B.name: creature_B.activations_history
    }
  }
