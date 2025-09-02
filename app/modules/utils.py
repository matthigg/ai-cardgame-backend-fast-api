import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from app.config import ACTION_NAMES, CONFIG, CREATURES

def create_state(creature, opponent):
  return torch.tensor([creature.hp, creature.energy, opponent.hp, opponent.energy], dtype=torch.float32)

def choose_action(nn_model, state, eps): 
  logits = nn_model(state)
  probs = F.softmax(logits, dim=0)
  if np.random.rand() < eps:
    action_idx = np.random.randint(len(ACTION_NAMES))
  else:
    dist = torch.distributions.Categorical(probs)
    action_idx = dist.sample().item()
  return action_idx, probs

def create_checkpoint_paths(creature_A, creature_B):
  A_id = f"checkpoint_{creature_A.name}_{CREATURES[creature_A.name]['id']}.pt"
  B_id = f"checkpoint_{creature_B.name}_{CREATURES[creature_B.name]['id']}.pt"
  A_path = f"{CONFIG['checkpoint_dir']}/{A_id}.pt"
  B_path = f"{CONFIG['checkpoint_dir']}/{B_id}.pt"
  return A_path, B_path
