import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from app.config import ACTION_NAMES, CHECKPOINT_DIR, PLAYERS_DIR

os.makedirs(PLAYERS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

def get_player_json_path(player_name: str):
  return os.path.join(PLAYERS_DIR, f"{player_name.lower()}.json")

def get_checkpoint_path(creature_id: int):
  return os.path.join(CHECKPOINT_DIR, f"{creature_id}.pt")
