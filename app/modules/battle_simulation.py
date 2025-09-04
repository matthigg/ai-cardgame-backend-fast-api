import torch
import random
from app.config import ACTION_NAMES, CONFIG
from app.modules.logging_utils import append_battle_log
from app.modules.utils import choose_action, create_state

def simulate_battle(creature_A, creature_B, epoch, max_ticks, epsilons):
  epsilon_A, epsilon_B = epsilons
  creature_A.reset()
  creature_B.reset()

  battle_log = []
  rewards = {creature_A.name: 0.0, creature_B.name: 0.0}
  zero = torch.zeros(len(ACTION_NAMES))

  def abl_zero_reward(actor, target, message):
    """Append zero-reward events like STUNNED, KNOCKOUT, POISONED, or STALEMATE."""
    append_battle_log(epoch, tick, actor, target, battle_log, message, zero, -1, 0.0)

  def check_for_knockouts():
    if not creature_A.is_alive() or not creature_B.is_alive():
      if creature_A.is_alive():
        abl_zero_reward(creature_A, creature_B, '*KNOCKOUT*')
      elif creature_B.is_alive():
        abl_zero_reward(creature_B, creature_A, '*KNOCKOUT*')
      return finalize_battle(creature_A, creature_B, rewards, battle_log)

  for tick in range(max_ticks):
    result = check_for_knockouts()
    if result:
      return result
    
    # Determine turn order each tick
    creatures = [creature_A, creature_B]
    creatures.sort(key=lambda c: c.speed, reverse=True)
    if creatures[0].speed == creatures[1].speed and random.random() < 0.5:
      creatures[0], creatures[1] = creatures[1], creatures[0]
    turn_order = creatures

    for creature in turn_order:
      opponent = creature_B if creature is creature_A else creature_A
      epsilon = epsilon_A if creature is creature_A else epsilon_B

      # Process statuses
      creature.process_statuses(opponent, abl_zero_reward)
      result = check_for_knockouts()
      if result:
        return result

      if 'stun' in creature.statuses:
        abl_zero_reward(creature, opponent, '*STUNNED*')
        continue

      action_index, probs = choose_action(creature.nn, create_state(creature, opponent), epsilon)
      action_name, action_fn = creature.actions[action_index]

      if action_name not in ['attack', 'defend', 'recover']:
        reward = action_fn(opponent, action_name)
      else:
        reward = action_fn(opponent)

      rewards[creature.name] += reward

      # If action caused knockout
      if not opponent.is_alive():
        abl_zero_reward(creature, opponent, '*KNOCKOUT*')
        return finalize_battle(creature_A, creature_B, rewards, battle_log)

      append_battle_log(
        epoch, tick,
        creature,
        opponent,
        battle_log,
        action_name,
        probs,
        action_index,
        reward
      )

  # Stalemate
  abl_zero_reward(creature_A, creature_B, '*STALEMATE*')
  abl_zero_reward(creature_B, creature_A, '*STALEMATE*')
  return finalize_battle(creature_A, creature_B, rewards, battle_log, stalemate=True)


def finalize_battle(creature_A, creature_B, rewards, battle_log, stalemate=False):
  """Determine winner, apply rewards, and finalize log ordering."""
  reward_win_A = creature_A.reward_config.get('win', CONFIG['reward_win'])
  reward_lose_A = creature_A.reward_config.get('lose', CONFIG['reward_lose'])
  reward_win_B = creature_B.reward_config.get('win', CONFIG['reward_win'])
  reward_lose_B = creature_B.reward_config.get('lose', CONFIG['reward_lose'])

  winner = None
  if stalemate:
    winner = "stalemate"
  elif creature_A.hp > creature_B.hp:
    rewards[creature_A.name] += reward_win_A
    rewards[creature_B.name] += reward_lose_B
    winner = creature_A.name
  elif creature_B.hp > creature_A.hp:
    rewards[creature_B.name] += reward_win_B
    rewards[creature_A.name] += reward_lose_A
    winner = creature_B.name

  if not stalemate:
    for entry in reversed(battle_log):
      if entry['creature'] == creature_A.name:
        entry['reward'] += rewards[creature_A.name]
        break
    for entry in reversed(battle_log):
      if entry['creature'] == creature_B.name:
        entry['reward'] += rewards[creature_B.name]
        break

  if CONFIG['sort_logs_by_creature']:
    battle_log.sort(key=lambda x: (x['creature'], x['epoch'], x['tick']))
    
  return rewards[creature_A.name], rewards[creature_B.name], battle_log, winner
