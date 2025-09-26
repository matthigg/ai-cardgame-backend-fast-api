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
    raise HTTPException(status_code=404, detail="Player not found")

  # Assuming unique player names => first match is the player
  with open(matches[0], "r") as f:
    data = json.load(f)
  return data


# ----------  2) Create Player ----------
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
      detail=f"Invalid creature '{creature}'. Use /players/creature-templates for valid names."
    )

  # Ensure no duplicate player
  pattern = os.path.join(PLAYERS_DIR, f"{name}_*.json")
  if glob.glob(pattern):
    raise HTTPException(status_code=409, detail="Player name already exists")

  player, player_path = create_player(name, [creature])

  with open(player_path, "r") as f:
    data = json.load(f)
  return data


# ----------  3) Get Creature Templates ----------
@router.get("/creature-templates")
def get_creature_templates():
  """
  Returns the entire CREATURE_TEMPLATES object
  so the UI can display available starter creatures.
  """
  return CREATURE_TEMPLATES
