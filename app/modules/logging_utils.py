import os
from app.config import CONFIG
from app.modules.utils import create_state

# ------------------ Append Battle Log ------------------

def append_battle_log(epoch, tick, creature, opponent, battle_log, action_name, probs, action_idx, reward):
  battle_log.append({
    'epoch': epoch,
    'tick': tick,
    'creature': creature.name,
    'state': create_state(creature, opponent).detach().numpy().tolist(),
    'action': action_name,
    'action_idx': int(action_idx),
    'probs': probs.detach().numpy().tolist(),
    'hp': creature.hp,
    'energy': creature.energy,
    'statuses': creature.statuses.copy(),
    'reward': reward,
  })

# ------------------ Logging ------------------

def write_logs(batched_logs, last_epoch, finalLog, finalWins=None):
  start_epoch = batched_logs[0][0] if batched_logs else 0
  end_epoch = batched_logs[-1][0] if batched_logs else 0
  filename = os.path.join(CONFIG['log_dir'], f'battle_log_{start_epoch:04d}_{end_epoch:04d}.txt')
  filenameFinal = os.path.join(CONFIG['log_dir'], 'summary.txt')

  # Write normal logs
  if not finalLog and CONFIG['write_battle_logs']:
    with open(filename, 'w') as f:
      for epoch, battle_log, reward_A, reward_B, wins_A, wins_B in batched_logs:
        f.write(f"Epoch {epoch}\n")
        f.write(f"Wins - {wins_A} | {wins_B}\n\n")
        for entry in battle_log:
          f.write(f"{entry['tick']:3} | {entry['creature']} | {entry['action']:10} "
                  f"{[f'{p:.2f}' for p in entry['probs']]} {entry['hp']:3} | {entry['energy']:3} "
                  f"| {entry['reward']:5.2f} | {entry['statuses']}\n")
        f.write('\n')

  # Write final summary log
  if finalLog and finalWins and CONFIG['write_battle_summary_log']:
    creature_names = list(finalWins.keys())
    epoch_batch_size = CONFIG['epoch_batch_size']

    # Initialize stats dynamically
    total_stats = {c: {'attack':0,'defend':0,'poison':0,'stun':0,'recover':0,'knockout':0,'stunned':0,'poisoned':0} 
                   for c in creature_names}

    # Aggregate stats from all batched logs
    for epoch, battle_log, _, _, _, _ in batched_logs:
      for entry in battle_log:
        c = entry['creature']
        action = entry['action']
        if action in ['attack','defend','recover','poison','stun']:
          total_stats[c][action] += 1
        elif action == '*KNOCKOUT*':
          total_stats[c]['knockout'] += 1
        elif action == '*STUNNED*':
          total_stats[c]['stunned'] += 1
        elif action == '*POISONED*':
          total_stats[c]['poisoned'] += 1

    with open(filenameFinal, 'w') as f:
      for c in creature_names:
        f.write(f"---------------------------------------------------------------\n")
        f.write(f"{c} | Total Wins: {finalWins[c]} | Avg Wins: {finalWins[c]/epoch_batch_size:.0%}\n")
        f.write(f"---------------------------------------------------------------\n")
        f.write(f"  Attack: {total_stats[c]['attack']}\n")
        f.write(f"  Defend: {total_stats[c]['defend']}\n")
        f.write(f"  Poison: {total_stats[c]['poison']}\n")
        f.write(f"  Stun: {total_stats[c]['stun']}\n")
        f.write(f"  Recover: {total_stats[c]['recover']}\n")
        f.write(f"  KO: {total_stats[c]['knockout']}\n")
        f.write(f"  Stun: {total_stats[c]['stunned']}\n")
        f.write(f"  Poison: {total_stats[c]['poisoned']}\n\n")
      f.write(f"---------------------------------------------------------------\n")
      f.write(f"Epoch Batch Size: {epoch_batch_size} | Total Epochs: {last_epoch}\n")
      f.write(f"---------------------------------------------------------------\n")
