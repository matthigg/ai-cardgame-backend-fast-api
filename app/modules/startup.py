# app/modules/startup.py
from app.modules.player_factory import create_player

def bootstrap_players():

  print('=============== BOOTSTRAP ===============')
  
  # Example: create Alice with Bear, Bob with Snake
  create_player("Alice", ["Bear"])
  create_player("Bob", ["Snake"])
  print("✅ Players bootstrapped successfully.")
