# app/api/player_routes.py
from fastapi import APIRouter
from app.modules.player_manager import add_active_player, remove_active_player, list_active_players
from app.modules.creature_manager import load_creature, add_active_creature, _active_creatures
from app.modules.creature_manager import init_creatures  # for nn_model template

router = APIRouter()

@router.post("/login")
def login_player(name: str, player_id: int):
  """
  Log in or create a player.
  Loads from /players if file exists, otherwise creates new.
  Ensures that all player's creatures are loaded from persistent files.
  """
  player = add_active_player(name, player_id)
  if not player:
    return {"error": "Unable to create or load player"}

  # Ensure all creatures for this player are loaded
  for i, creature in enumerate(player.creatures):
    key = f"{creature.id}_{creature.name}"
    if key not in _active_creatures:
      # Create a fresh NN model based on template
      template_stats = creature.__dict__  # keep existing stats as fallback
      nn_model = init_creatures({creature.name: template_stats})[0][creature.name].nn
      loaded = load_creature(creature.id, creature.name, nn_model)
      if loaded:
        player.creatures[i] = loaded
        add_active_creature(loaded)

  return {
    "message": f"Player {name} ({player_id}) active",
    "player": player.to_dict()
  }

@router.post("/logout")
def logout_player(name: str, player_id: int):
  remove_active_player(name, player_id)
  return {"message": f"Player {name} ({player_id}) logged out"}

@router.get("/active")
def active_players():
  return {"active_players": list_active_players()}
