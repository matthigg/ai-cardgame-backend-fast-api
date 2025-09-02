import torch
from app.config import ACTION_NAMES, CONFIG
from app.modules.logging_utils import append_battle_log
from app.modules.utils import choose_action, create_state

# ------------------ Battle Simulation ------------------

def simulate_battle(creature_A, creature_B, epoch, batch_size, epsilon):
  creature_A.hp = creature_A.max_hp
  creature_A.energy = creature_A.max_energy
  creature_B.hp = creature_B.max_hp
  creature_B.energy = creature_B.max_energy
  battle_log = []
  creatures = [creature_A, creature_B]
  rewards = {creature_A.name: 0.0, creature_B.name: 0.0}
  turn_order = sorted(creatures, key=lambda c: c.speed, reverse=True)
  zero = torch.zeros(len(ACTION_NAMES)) # no probabilities for Stun or KO

  for tick in range(batch_size):
    for creature in turn_order:

      opponent = creature_B if creature is creature_A else creature_A
      if not creature.is_alive() or not opponent.is_alive():
        continue

      creature.process_statuses(epoch, tick, creature, opponent, battle_log, zero)
      if not creature.is_alive() or not opponent.is_alive():
        continue

      # Skip turn if stunned
      if 'Stun' in creature.statuses:
        append_battle_log(epoch, tick, creature, opponent, battle_log, '*STUNNED*', zero, -1, 0.0)
        continue

      # Choose and perform action, assign per-action reward
      action_idx, probs = choose_action(creature.nn, create_state(creature, opponent), epsilon)
      action_name, action_fn = creature.actions[action_idx]

      # If it's a special, pass the action name to use_special
      if action_name in creature.special_abilities:
        reward = action_fn(opponent, action_name)
      else:
        reward = action_fn(opponent)
      rewards[creature.name] += reward

      if opponent.is_alive():
        append_battle_log(epoch, tick, creature, opponent, battle_log, action_name, probs, action_idx, reward)
      else:

        # Swap opponent and creature in order to log opponent with *KNOCKOUT* message
        append_battle_log(epoch, tick, opponent, creature, battle_log, '*KNOCKOUT*', zero, -1, 0.0)

  # Final win/loss rewards
  winner = None
  if creature_A.hp > creature_B.hp:
    rewards[creature_A.name] += CONFIG['reward_win']
    rewards[creature_B.name] += CONFIG['reward_lose']
    winner = creature_A.name
  elif creature_B.hp > creature_A.hp:
    rewards[creature_B.name] += CONFIG['reward_win']
    rewards[creature_A.name] += CONFIG['reward_lose']
    winner = creature_B.name

  # Apply final rewards to last entries
  for entry in reversed(battle_log):
    if entry['creature'] == creature_A.name:
      entry['reward'] += rewards[creature_A.name]
      break
  for entry in reversed(battle_log):
    if entry['creature'] == creature_B.name:
      entry['reward'] += rewards[creature_B.name]
      break

  battle_log.sort(key=lambda x: (x['creature'], x['epoch'], x['tick']))
  return rewards[creature_A.name], rewards[creature_B.name], battle_log, winner
