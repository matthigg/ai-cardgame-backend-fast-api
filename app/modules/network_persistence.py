import os
import torch
from app.config import CONFIG
from app.modules.creature_manager import Creature
from app.modules.utils import create_checkpoint_path

def save_checkpoints(creature_A: Creature, creature_B: Creature, optimizer_A, optimizer_B):
  os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

  A_path = create_checkpoint_path(creature_A)
  B_path = create_checkpoint_path(creature_B)

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

def resume_from_checkpoint(creature_A: Creature, creature_B: Creature, optimizer_A, optimizer_B):
  A_path = create_checkpoint_path(creature_A)
  B_path = create_checkpoint_path(creature_B)

  if os.path.exists(A_path):
    checkpoint_A = torch.load(A_path)
    creature_A.nn.load_state_dict(checkpoint_A['model_state_dict'])
    optimizer_A.load_state_dict(checkpoint_A['optimizer_state_dict'])
    creature_A.activations_history = checkpoint_A.get('activations_history', [])
    creature_A.current_epoch = checkpoint_A.get('epoch', 0)

  if os.path.exists(B_path):
    checkpoint_B = torch.load(B_path)
    creature_B.nn.load_state_dict(checkpoint_B['model_state_dict'])
    optimizer_B.load_state_dict(checkpoint_B['optimizer_state_dict'])
    creature_B.activations_history = checkpoint_B.get('activations_history', [])
    creature_B.current_epoch = checkpoint_B.get('epoch', 0)
