# app/modules/startup.py
import os
import glob
import json
from app.modules.player_manager import create_player
from app.config import GENERATED_DIR, NPCS_DIR

NPCS = [
  {"name": "NPC_Warrior", "creatures": ["Bear", "Snake"]},
  {"name": "NPC_Rogue",   "creatures": ["Snake"]},
  {"name": "NPC_Mage",    "creatures": ["Bear"]},
]

def bootstrap_players():
  pass
  
  # Example: create Alice with Bear, Bob with Snake
  # create_player("Alice", ["Bear"])
  # create_player("Bob", ["Snake"])
  # print("✅ Players bootstrapped successfully.")

def bootstrap_npcs():
  os.makedirs(os.path.join(GENERATED_DIR, NPCS_DIR), exist_ok=True)

  for npc in NPCS:
    # Skip if NPC already exists (idempotent)
    pattern = os.path.join(GENERATED_DIR, NPCS_DIR, f"{npc['name']}_*.json")
    if glob.glob(pattern):
      continue

    # Create with all listed creatures
    player, player_path = create_player(npc["name"], npc["creatures"])
    print(f"✅ NPC created: {npc['name']} -> {player_path}")
