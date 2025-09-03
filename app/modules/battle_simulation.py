import torch
from app.config import ACTION_NAMES, CONFIG
from app.modules.logging_utils import append_battle_log
from app.modules.utils import choose_action, create_state

def simulate_battle(creature_A, creature_B, epoch, batch_size, epsilons):
  epsilon_A, epsilon_B = epsilons
  creature_A.hp, creature_A.energy = creature_A.max_hp, creature_A.max_energy
  creature_B.hp, creature_B.energy = creature_B.max_hp, creature_B.max_energy
  battle_log = []
  rewards = {creature_A.name: 0.0, creature_B.name: 0.0}
  turn_order = sorted([creature_A, creature_B], key=lambda c: c.speed, reverse=True)
  zero = torch.zeros(len(ACTION_NAMES))

  for tick in range(batch_size):
    for creature in turn_order:
      opponent = creature_B if creature is creature_A else creature_A
      epsilon = epsilon_A if creature is creature_A else epsilon_B

      if not creature.is_alive() or not opponent.is_alive():
        continue

      creature.process_statuses(epoch, tick, creature, opponent, battle_log, zero)
      if not creature.is_alive() or not opponent.is_alive():
        continue

      if 'stun' in creature.statuses:
        append_battle_log(epoch, tick, creature, opponent, battle_log, '*STUNNED*', zero, -1, 0.0)
        continue

      action_idx, probs = choose_action(creature.nn, create_state(creature, opponent), epsilon)
      action_name, action_fn = creature.actions[action_idx]

      if action_name not in ['attack', 'defend', 'recover']:
        reward = action_fn(opponent, action_name)
      else:
        reward = action_fn(opponent)

      rewards[creature.name] += reward
      append_battle_log(epoch, tick, creature if opponent.is_alive() else opponent, opponent if opponent.is_alive() else creature, battle_log, action_name if opponent.is_alive() else '*KNOCKOUT*', probs, action_idx, reward)

  reward_win_A = creature_A.reward_config.get('win', CONFIG['reward_win'])
  reward_lose_A = creature_A.reward_config.get('lose', CONFIG['reward_lose'])
  reward_win_B = creature_B.reward_config.get('win', CONFIG['reward_win'])
  reward_lose_B = creature_B.reward_config.get('lose', CONFIG['reward_lose'])

  winner = None
  if creature_A.hp > creature_B.hp:
    rewards[creature_A.name] += reward_win_A
    rewards[creature_B.name] += reward_lose_B
    winner = creature_A.name
  elif creature_B.hp > creature_A.hp:
    rewards[creature_B.name] += reward_win_B
    rewards[creature_A.name] += reward_lose_A
    winner = creature_B.name

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
