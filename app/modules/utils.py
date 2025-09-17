import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from app.config import ACTION_NAMES, CONFIG

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

def create_checkpoint_path(creature):
  file_name = f"checkpoint_{creature.name}_{creature.id}"
  complete_path = f"{CONFIG['checkpoint_dir']}/{file_name}.pt"
  return complete_path
