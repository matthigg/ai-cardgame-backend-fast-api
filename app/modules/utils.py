import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from app.config import ACTION_NAMES, BATTLE_LOGS_DIR, CHECKPOINT_DIR, GENERATED_DIR, NPCS_DIR, PLAYERS_DIR

# Ensure root generated directory exists
os.makedirs(GENERATED_DIR, exist_ok=True)

# Ensure nested directories exist
os.makedirs(os.path.join(GENERATED_DIR, PLAYERS_DIR), exist_ok=True)
os.makedirs(os.path.join(GENERATED_DIR, CHECKPOINT_DIR), exist_ok=True)
os.makedirs(os.path.join(GENERATED_DIR, BATTLE_LOGS_DIR), exist_ok=True)

def create_state(creature, opponent, tick=None, max_ticks=None):
  """Return normalized feature vector describing the current battle state."""

  def normalize(value, max_value):
    try:
      # handle both callable methods and numeric values
      val = value() if callable(value) else value
      return val / max_value if max_value > 0 else 0.0
    except Exception:
      return 0.0

  state = [
    # --- Creature (self) ---
    normalize(creature.hp, creature.max_hp),
    normalize(creature.energy, creature.max_energy),
    normalize(creature.attack, 100),
    normalize(creature.defend, 100),
    normalize(creature.speed, 100),
    1.0 if 'poison' in creature.statuses else 0.0,
    1.0 if 'stun' in creature.statuses else 0.0,

    # --- Opponent ---
    normalize(opponent.hp, opponent.max_hp),
    normalize(opponent.energy, opponent.max_energy),
    normalize(opponent.attack, 100),
    normalize(opponent.defend, 100),
    normalize(opponent.speed, 100),
    1.0 if 'poison' in opponent.statuses else 0.0,
    1.0 if 'stun' in opponent.statuses else 0.0,
  ]

  # Optional: include normalized tick progress
  if tick is not None and max_ticks:
    state.append(normalize(tick, max_ticks))

  return torch.tensor(state, dtype=torch.float32)


def choose_action(nn_model, state, eps):
  logits = nn_model(state)
  probs = F.softmax(logits, dim=0)
  if np.random.rand() < eps:
    action_idx = np.random.randint(len(ACTION_NAMES))
  else:
    dist = torch.distributions.Categorical(probs)
    action_idx = dist.sample().item()
  return action_idx, probs

def get_player_json_path(player_name: str, player_id: int):
  return os.path.join(
    GENERATED_DIR,
    PLAYERS_DIR,
    f"{player_name.lower()}_{player_id}.json"
  )

def get_npc_json_path(name: str, npc_id: int) -> str:
  return os.path.join(
    GENERATED_DIR, 
    NPCS_DIR, 
    f"{name}_{npc_id}.json"
  )

def get_checkpoint_path(player_name: str, player_id: int, creature_name: str, creature_id: int):
  """
  Create a unique checkpoint path for a creature belonging to a player.
  Example: generated/checkpoints/alice_1_dragon_2.pt
  """
  filename = f"{player_name.lower()}_{player_id}_{creature_name.lower()}_{creature_id}.pt"
  return os.path.join(GENERATED_DIR, CHECKPOINT_DIR, filename)
