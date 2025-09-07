import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from app.config import ACTION_NAMES, CONFIG, CREATURE_TEMPLATES

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
  A_id = f"checkpoint_{creature_A.name}_{CREATURE_TEMPLATES[creature_A.name]['id']}"
  B_id = f"checkpoint_{creature_B.name}_{CREATURE_TEMPLATES[creature_B.name]['id']}"
  A_path = f"{CONFIG['checkpoint_dir']}/{A_id}.pt"
  B_path = f"{CONFIG['checkpoint_dir']}/{B_id}.pt"
  return A_path, B_path

def create_checkpoint_paths_by_name(creature_name_a: str = 'A', creature_name_b: str = 'B') -> tuple[str, str]:
  A_id = f"checkpoint_{creature_name_a}_{CREATURE_TEMPLATES[creature_name_a]['id']}"
  B_id = f"checkpoint_{creature_name_b}_{CREATURE_TEMPLATES[creature_name_b]['id']}"
  A_path = f"{CONFIG['checkpoint_dir']}/{A_id}.pt"
  B_path = f"{CONFIG['checkpoint_dir']}/{B_id}.pt"
  return A_path, B_path
