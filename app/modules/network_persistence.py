import os
import torch
from app.config import CONFIG, CREATURES
from app.modules.utils import create_checkpoint_paths

# ------------------ Network Persistence ------------------

def create_checkpoint_file(checkpoint_path, creature, optimizer):
  print(f"⚠️ Missing checkpoint: {checkpoint_path}. Creating new one...")
  os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
  torch.save({
    'epoch': 0,
    'model_state_dict': creature.nn.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
  }, checkpoint_path)
  print(f"✅ Created fresh checkpoint for {creature.name} at epoch 0: {checkpoint_path}")

def save_checkpoint(checkpoint_path, creature, optimizer):
  checkpoint = torch.load(checkpoint_path)
  last_epoch = checkpoint.get('epoch', 0)
  torch.save({
    'epoch': last_epoch + CONFIG['epoch_batch_size'],
    'model_state_dict': creature.nn.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
  }, checkpoint_path)
  print(f"💾 Saved checkpoint for {creature.name} at epoch {last_epoch}: {checkpoint_path}")

def load_checkpoint(checkpoint_path, creature, optimizer, config_key=None):
  if not os.path.isfile(checkpoint_path):
    create_checkpoint_file(checkpoint_path, creature, optimizer)
    return 0
  checkpoint = torch.load(checkpoint_path)
  creature.nn.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  last_epoch = checkpoint.get('epoch', 0)
  print(f"📂 Resumed {creature.name} from {checkpoint_path} at epoch {last_epoch}")
  return last_epoch

def save_checkpoints(creature_A, creature_B, optimizer_A, optimizer_B):
  A_path, B_path = create_checkpoint_paths(creature_A, creature_B)
  save_checkpoint(A_path, creature_A, optimizer_A)
  save_checkpoint(B_path, creature_B, optimizer_B)

def resume_from_checkpoint(creature_A, creature_B, optimizer_A, optimizer_B):
  A_path, B_path = create_checkpoint_paths(creature_A, creature_B)
  last_epoch_A = load_checkpoint(A_path, creature_A, optimizer_A)
  last_epoch_B = load_checkpoint(B_path, creature_B, optimizer_B)
  print("🔄 Checkpoint summary:")
  print(f"  {creature_A.name} -> {A_path} (next epoch: {last_epoch_A})")
  print(f"  {creature_B.name} -> {B_path} (next epoch: {last_epoch_B})")
