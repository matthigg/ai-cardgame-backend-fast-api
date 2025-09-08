# app/api/player_routes.py
from fastapi import APIRouter
from app.modules.player_manager import add_active_player, remove_active_player, list_active_players

router = APIRouter()

@router.post("/login")
def login_player(name: str, player_id: int):
  """
  Log in or create a player.
  Loads from /players if file exists, otherwise creates new.
  """
  player = add_active_player(name, player_id)
  if not player:
    return {"error": "Unable to create or load player"}
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
