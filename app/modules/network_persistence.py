import os
import torch
from app.config import CONFIG

# ------------------ Network Persistence ------------------

def save_checkpoint(creature, optimizer, epoch, log_dir):
  os.makedirs(log_dir, exist_ok=True)
  filename = f"nn_{creature.name}.pt"
  path = os.path.join(log_dir, filename)

  # Save the "next epoch" so resuming is correct
  torch.save({
    'epoch': epoch + 1, # store the next epoch
    'model_state_dict': creature.nn.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
  }, path)
  print(f"💾 Saved checkpoint for {creature.name} at epoch {epoch}: {path}")
  return path


def load_checkpoint(creature, optimizer, checkpoint_path, config_key=None):
  if not os.path.isfile(checkpoint_path):
    print(f"⚠️ Missing checkpoint: {checkpoint_path}. Creating new one...")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
      'epoch': 0,
      'model_state_dict': creature.nn.state_dict(),
      'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"✅ Created fresh checkpoint for {creature.name} at epoch 0: {checkpoint_path}")

    if config_key:
      CONFIG[config_key] = checkpoint_path
      print(f"🔧 Updated CONFIG['{config_key}'] = {checkpoint_path}")

    return 0

  checkpoint = torch.load(checkpoint_path)
  creature.nn.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  last_epoch = checkpoint.get('epoch', 0)

  print(f"📂 Resumed {creature.name} from {checkpoint_path} at epoch {last_epoch}")
  return last_epoch

def save_best_checkpoint(epoch, best_reward_A, best_reward_B, reward_A, reward_B,
                        creature_A, creature_B, optimizer_A, optimizer_B):
  """
  Save the best performing model for each creature if its reward exceeds previous best.
  Updates CONFIG paths for resuming.
  """
  if reward_A > best_reward_A:
    best_reward_A = reward_A
    path_A = save_checkpoint(creature_A, optimizer_A, epoch, CONFIG['checkpoint_dir'])
    CONFIG['resume_from_checkpoint_A'] = path_A  # store path for future resuming

  if reward_B > best_reward_B:
    best_reward_B = reward_B
    path_B = save_checkpoint(creature_B, optimizer_B, epoch, CONFIG['checkpoint_dir'])
    CONFIG['resume_from_checkpoint_B'] = path_B  # store path for future resuming

  return best_reward_A, best_reward_B


def resume_from_checkpoint(creature_A, creature_B, optimizer_A, optimizer_B):
  """
  Resume training from saved checkpoints for two creatures.
  Returns the start epochs per creature.
  """
  ckpt_A = f"checkpoints/nn_{creature_A.name}.pt"
  ckpt_B = f"checkpoints/nn_{creature_B.name}.pt"

  start_epoch_A = load_checkpoint(creature_A, optimizer_A, ckpt_A)
  start_epoch_B = load_checkpoint(creature_B, optimizer_B, ckpt_B)

  # Summary print
  print("🔄 Checkpoint summary:")
  print(f"  {creature_A.name} -> {ckpt_A} (next epoch: {start_epoch_A})")
  print(f"  {creature_B.name} -> {ckpt_B} (next epoch: {start_epoch_B})")

  return {
    creature_A.name: start_epoch_A,
    creature_B.name: start_epoch_B
  }


