import json
import os
from app.config import BATTLE_LOGS_DIR, CONFIG, GENERATED_DIR
from app.modules.utils import create_state

# ------------------ Append Battle Log ------------------

def append_battle_log(epoch, tick, creature, opponent, battle_log, action_name, probs, action_idx, reward):
  """Append a single battle event to the log."""
  data = {
    'epoch': epoch,
    'tick': tick,
    'creature': creature.name,
    'state': create_state(creature, opponent).detach().numpy().tolist(),
    'action': action_name,
    'action_idx': int(action_idx),
    'probs': probs.detach().numpy().tolist() if hasattr(probs, 'detach') else list(probs),
    'hp': creature.hp,
    'energy': creature.energy,
    'statuses': creature.statuses.copy(),
    'reward': reward,
    'owner': creature.owner if hasattr(creature, 'owner') else 'unknown'
  }
  battle_log.append(data)

# ------------------ Batched Logging ------------------

def write_logs(batched_logs, last_epochs, finalLog, final_wins=None):
  """Write batched logs or final summary to disk."""
  start_epoch = batched_logs[0][0] if batched_logs else 0
  end_epoch = batched_logs[-1][0] if batched_logs else 0
  filename = os.path.join(GENERATED_DIR, BATTLE_LOGS_DIR, f'battle_log_{start_epoch:04d}_{end_epoch:04d}.txt')
  filenameFinal = os.path.join(GENERATED_DIR, BATTLE_LOGS_DIR, 'summary.txt')

  # Write normal logs
  if not finalLog and CONFIG['write_battle_logs']:
    with open(filename, 'w') as f:
      for epoch, battle_log, reward_A, reward_B, wins_A, wins_B in batched_logs:
        f.write(f"Epoch {epoch}\n")
        f.write(f"Wins - {wins_A} | {wins_B}\n\n")
        for entry in battle_log:
          f.write(
            f"{entry['tick']:3} | {entry['creature']:7} | {entry['action']:11} "
            f"{entry['hp']:3} | {entry['energy']:3} | {entry['reward']:5.2f} | "
            f"{str(entry['statuses']):14} {[f'{p:.2f}' for p in entry['probs']]}\n"
          )
        f.write('\n')

  summary_data = None

  # Write final summary log
  if finalLog and final_wins and CONFIG['write_battle_summary_log']:
    epoch_batch_size = CONFIG['epoch_batch_size']

    # Gather all observed (creature_name, owner_name) pairs
    observed_pairs = set()
    for epoch, battle_log, _, _, _, _ in batched_logs:
      for entry in battle_log:
        name = entry.get('creature', 'unknown')
        owner = entry.get('owner', 'unknown')
        observed_pairs.add((name, owner))

    # Initialize stats for each observed pair
    total_stats = {}
    for name, owner in observed_pairs:
      key = f"{name} ({owner})"
      total_stats[key] = {
        'attack': 0, 'defend': 0, 'poison': 0, 'stun': 0,
        'recover': 0, 'knockout': 0, 'stunned': 0,
        'poisoned': 0, 'stalemates': 0
      }

    # Map raw actions to stat keys
    action_map = {
      'attack': 'attack',
      'defend': 'defend',
      'recover': 'recover',
      'poison': 'poison',
      'stun': 'stun',
      '*KNOCKOUT*': 'knockout',
      '*STUNNED*': 'stunned',
      '*POISONED*': 'poisoned',
      '*STALEMATE*': 'stalemates'
    }

    # Count actions per (name, owner)
    for epoch, battle_log, _, _, _, _ in batched_logs:
      for entry in battle_log:
        name = entry.get('creature', 'unknown')
        owner = entry.get('owner', 'unknown')
        key = f"{name} ({owner})"

        if key not in total_stats:
          total_stats[key] = {k: 0 for k in ['attack','defend','poison','stun','recover','knockout','stunned','poisoned','stalemates']}

        action_raw = entry.get('action', '')
        stat_key = action_map.get(action_raw, None)
        if stat_key:
          total_stats[key][stat_key] += 1

    # Build summary_data
    summary_data = {}
    for name, owner in observed_pairs:
      key = f"{name} ({owner})"
      total_wins = final_wins.get(name, 0)
      total_epochs = last_epochs.get(name, 0)

      summary_data[key] = {
        "name": name,
        "owner": owner,
        "totalWins": total_wins,
        "avgWins": total_wins / epoch_batch_size if epoch_batch_size else 0,
        "totalEpochs": total_epochs,
        "stats": total_stats.get(key, {k: 0 for k in ['attack','defend','poison','stun','recover','knockout','stunned','poisoned','stalemates']})
      }

    # Write JSON file
    filenameJson = os.path.join(GENERATED_DIR, BATTLE_LOGS_DIR, 'summary.json')
    with open(filenameJson, 'w') as fjson:
      json.dump(summary_data, fjson, indent=2)

    # Write final text summary
    with open(filenameFinal, 'w') as f:
      for key, data in summary_data.items():
        f.write("---------------------------------------------------------------\n")
        f.write(
          f"{key} | Total Wins: {data['totalWins']} | "
          f"Avg Wins: {data['avgWins']:.0%} | Total Epochs: {data['totalEpochs']}\n"
        )
        f.write("---------------------------------------------------------------\n")
        for stat_name, stat_val in data['stats'].items():
          f.write(f"  {stat_name.capitalize():10}: {stat_val}\n")
        f.write("\n")
      f.write("---------------------------------------------------------------\n")
      f.write(f"Epoch Batch Size: {epoch_batch_size}\n")
      f.write("---------------------------------------------------------------\n")

  return summary_data
