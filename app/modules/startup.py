# app/modules/startup.py
import os
import glob
from app.config import GENERATED_DIR, NPCS_DIR
from app.modules.player_manager import create_npc

NPCS = [
  {"name": "npc_warrior", "creatures": ["Bear", "Snake"]},
  {"name": "npc_rogue",   "creatures": ["Snake"]},
  {"name": "npc_mage",    "creatures": ["Bear"]},
]

def bootstrap_players():
  pass
  
  # Example: create Alice with Bear, Bob with Snake
  # create_player("Alice", ["Bear"])
  # create_player("Bob", ["Snake"])
  # print("✅ Players bootstrapped successfully.")

def bootstrap_npcs():
  """
  Create all configured NPCs if they don't already exist.
  Uses create_npc to place files in generated/npcs.
  """
  os.makedirs(os.path.join(GENERATED_DIR, NPCS_DIR), exist_ok=True)

  for npc in NPCS:
    # Skip if NPC already exists (idempotent)
    pattern = os.path.join(GENERATED_DIR, NPCS_DIR, f"{npc['name']}_*.json")
    if glob.glob(pattern):
      continue

    # Create NPC with all listed creatures
    npc_obj, npc_path = create_npc(npc["name"], npc["creatures"])
    print(f"✅ NPC created: {npc['name']} -> {npc_path}")
