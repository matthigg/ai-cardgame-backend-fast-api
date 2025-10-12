# app/routes/player_routes.py
import os
import glob
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.modules.player_manager import create_player
from app.config import CREATURE_TEMPLATES, GENERATED_DIR, PLAYERS_DIR

router = APIRouter(tags=["player"])

# ----------  Request Models ----------
class LoginRequest(BaseModel):
  name: str

class CreatePlayerRequest(BaseModel):
  name: str
  creature: str


# ----------  1) Login ----------
@router.post("/login")
def login_player(req: LoginRequest):
  """
  Attempt to find an existing player by name.
  Returns the player JSON if found, otherwise 404.
  """
  player_name = req.name.strip()
  if not player_name:
    raise HTTPException(status_code=400, detail="Player name cannot be empty")

  pattern = os.path.join(GENERATED_DIR, PLAYERS_DIR, f"{player_name}_*.json")
  matches = glob.glob(pattern)

  if not matches:
    raise HTTPException(status_code=404, detail=f"Player {player_name} not found")

  # Assuming unique player names => first match is the player
  with open(matches[0], "r") as f:
    data = json.load(f)
  return data

# ----------  2) Logout ----------
@router.post("/delete")
def delete_player(req: LoginRequest):
  """
  Delete the player's JSON file and all corresponding creature
  .pt files in generated/checkpoints.
  """
  player_name = req.name.strip()
  if not player_name:
    raise HTTPException(status_code=400, detail="Player name cannot be empty")

  # --- Delete player file ---
  player_pattern = os.path.join(GENERATED_DIR, PLAYERS_DIR, f"{player_name}_*.json")
  player_matches = glob.glob(player_pattern)
  if not player_matches:
    raise HTTPException(status_code=404, detail=f"Player {player_name} not found")

  for path in player_matches:
    try:
      os.remove(path)
    except Exception as e:
      raise HTTPException(status_code=500, detail=f"Failed to delete player file: {str(e)}")

  # --- Delete all creature checkpoints for this player ---
  checkpoint_dir = os.path.join(GENERATED_DIR, "checkpoints")
  checkpoint_pattern = os.path.join(checkpoint_dir, f"{player_name}_*.pt")
  checkpoint_matches = glob.glob(checkpoint_pattern)

  deleted_checkpoints = []
  for path in checkpoint_matches:
    try:
      os.remove(path)
      deleted_checkpoints.append(os.path.basename(path))
    except Exception as e:
      raise HTTPException(status_code=500, detail=f"Failed to delete checkpoint {path}: {str(e)}")

  return {
    "status": "success",
    "message": f"Player {player_name} and {len(deleted_checkpoints)} checkpoints deleted",
    "deleted_checkpoints": deleted_checkpoints,
  }


# ----------  3) Create Player ----------
@router.post("/create")
def create_new_player(req: CreatePlayerRequest):
  """
  Create a brand-new player with the given name and starting creature.
  Returns the created player JSON.
  """
  name = req.name.strip()
  creature = req.creature.strip()

  if not name:
    raise HTTPException(status_code=400, detail="Player name cannot be empty")
  if creature not in CREATURE_TEMPLATES:
    raise HTTPException(
      status_code=400,
      detail=f"Invalid creature '{creature}'."
    )

  # Ensure no duplicate player
  pattern = os.path.join(GENERATED_DIR, PLAYERS_DIR, f"{name}_*.json")
  if glob.glob(pattern):
    raise HTTPException(status_code=409, detail="Player name already exists")

  player, player_path = create_player(name, [creature])

  with open(player_path, "r") as f:
    data = json.load(f)
  return data


# ----------  4) Get Creature Templates ----------
@router.get("/creature-templates")
def get_creature_templates():
  """
  Returns the entire CREATURE_TEMPLATES object
  so the UI can display available starter creatures.
  """
  return CREATURE_TEMPLATES
