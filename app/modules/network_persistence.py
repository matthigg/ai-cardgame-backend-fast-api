import os
import torch
from app.modules.creature_manager import Creature
from app.modules.utils import get_checkpoint_path

def save_checkpoints(
  creature_A: Creature, creature_B: Creature,
  optimizer_A, optimizer_B,
  player_name_A: str, player_id_A: int,
  player_name_B: str, player_id_B: int
):
  """
  Save checkpoints for both creatures with optimizers.
  """
  A_path = get_checkpoint_path(player_name_A, player_id_A, creature_A.name, creature_A.id)
  B_path = get_checkpoint_path(player_name_B, player_id_B, creature_B.name, creature_B.id)

  torch.save({
    'epoch': getattr(creature_A, 'current_epoch', 0),
    'model_state_dict': creature_A.nn.state_dict(),
    'optimizer_state_dict': optimizer_A.state_dict(),
    'activations_history': getattr(creature_A, 'activations_history', [])
  }, A_path)

  torch.save({
    'epoch': getattr(creature_B, 'current_epoch', 0),
    'model_state_dict': creature_B.nn.state_dict(),
    'optimizer_state_dict': optimizer_B.state_dict(),
    'activations_history': getattr(creature_B, 'activations_history', [])
  }, B_path)

def resume_from_checkpoint(
  creature_A: Creature, creature_B: Creature,
  optimizer_A, optimizer_B,
  player_name_A: str, player_id_A: int,
  player_name_B: str, player_id_B: int
):
  """
  Resume training from checkpoints if available.
  Raises an error if optimizer state is missing.
  """
  A_path = get_checkpoint_path(player_name_A, player_id_A, creature_A.name, creature_A.id)
  B_path = get_checkpoint_path(player_name_B, player_id_B, creature_B.name, creature_B.id)

  if os.path.exists(A_path):
    checkpoint_A = torch.load(A_path)
    creature_A.nn.load_state_dict(checkpoint_A['model_state_dict'])
    if 'optimizer_state_dict' not in checkpoint_A or checkpoint_A['optimizer_state_dict'] is None:
      raise RuntimeError(f"Checkpoint for {creature_A.name} missing optimizer state at {A_path}")
    optimizer_A.load_state_dict(checkpoint_A['optimizer_state_dict'])
    creature_A.activations_history = checkpoint_A.get('activations_history', [])
    creature_A.current_epoch = checkpoint_A.get('epoch', 0)

  if os.path.exists(B_path):
    checkpoint_B = torch.load(B_path)
    creature_B.nn.load_state_dict(checkpoint_B['model_state_dict'])
    if 'optimizer_state_dict' not in checkpoint_B or checkpoint_B['optimizer_state_dict'] is None:
      raise RuntimeError(f"Checkpoint for {creature_B.name} missing optimizer state at {B_path}")
    optimizer_B.load_state_dict(checkpoint_B['optimizer_state_dict'])
    creature_B.activations_history = checkpoint_B.get('activations_history', [])
    creature_B.current_epoch = checkpoint_B.get('epoch', 0)
