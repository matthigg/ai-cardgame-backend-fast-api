import torch
import torch.nn as nn
from app.config import CONFIG

# ------------------ Neural Network ------------------

class NeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_sizes, output_size):
    super().__init__()
    layers = []
    last_size = input_size
    for h in hidden_sizes:
      layers.append(nn.Linear(last_size, h))
      layers.append(nn.ReLU())
      last_size = h
    layers.append(nn.Linear(last_size, output_size))
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

# ------------------ Reinforce Update ------------------

def reinforce_update(creature, optimizer, battle_log, baseline, entropy_beta=None):
  entropy_beta = entropy_beta or getattr(creature, 'nn_config', {}).get('entropy_beta', CONFIG['entropy_beta'])
  total_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
  optimizer.zero_grad()

  # Use creature-specific rewards for all actions, including special abilities
  reward_config = getattr(creature, 'reward_config', {})
  reward_map = {
    'attack': reward_config.get('attack', CONFIG['reward_attack']),
    'defend': reward_config.get('defend', CONFIG['reward_defend']),
    'recover': reward_config.get('recover', CONFIG['reward_recover']),
    'win': reward_config.get('win', CONFIG['reward_win']),
    'lose': reward_config.get('lose', CONFIG['reward_lose'])
  }
  # Add any special abilities defined in reward_config
  for ability_name in getattr(creature, 'special_abilities', []):
    reward_map[ability_name] = reward_config.get(ability_name, 0.01)

  for entry in battle_log:
    if entry['creature'] != creature.name or entry['action_idx'] == -1:
      continue
    probs = torch.tensor(entry['probs'], dtype=torch.float32) if not isinstance(entry['probs'], torch.Tensor) else entry['probs']
    action_idx_tensor = torch.tensor(int(entry['action_idx']), dtype=torch.long)
    dist = torch.distributions.Categorical(probs)
    log_prob = dist.log_prob(action_idx_tensor)

    # Reward comes from creature-specific reward_map, fallback to battle_log reward
    action_name = entry['action']
    reward = reward_map.get(action_name, entry['reward'])
    reward = torch.tensor(reward, dtype=torch.float32)

    loss = -log_prob * (reward - baseline)
    entropy = dist.entropy()
    loss -= entropy_beta * entropy
    total_loss = total_loss + loss

  total_loss.backward()
  optimizer.step()
