# app/routes/creature_routes.py
import os
import json
import glob
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.config import CREATURE_TEMPLATES, GENERATED_DIR, PLAYERS_DIR

router = APIRouter(tags=["creatures"])


# ---------- Request Model ----------
class RecruitRequest(BaseModel):
  template_name: str


# ---------- 1) Get Creature Templates ----------
@router.get("/templates")
def get_creature_templates():
  """
  Returns all creature templates defined in config.py.
  Used by the frontend recruitment page.
  """
  return CREATURE_TEMPLATES


# ---------- 2) Recruit Creature ----------
@router.post("/recruit/{player_name}")
def recruit_creature(player_name: str, req: RecruitRequest):
  """
  Adds a new creature (based on a template) to the player's JSON file.
  """
  template_name = req.template_name

  if template_name not in CREATURE_TEMPLATES:
    raise HTTPException(status_code=400, detail=f"Invalid creature template '{template_name}'.")

  # Locate player file
  player_pattern = os.path.join(GENERATED_DIR, PLAYERS_DIR, f"{player_name}_*.json")
  matches = glob.glob(player_pattern)
  if not matches:
    raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found.")

  player_path = matches[0]
  with open(player_path, "r") as f:
    player_data = json.load(f)

  # Build new creature object from template
  template = CREATURE_TEMPLATES[template_name]
  new_id = max([c["id"] for c in player_data["creatures"]] or [0]) + 1
  new_creature = {
    **template,
    "id": new_id,
    "owner": player_name,
    "runtime_state": {
      "hp": template["hp"],
      "energy": template["energy"],
      "statuses": {}
    }
  }

  # Append and save
  player_data["creatures"].append(new_creature)
  with open(player_path, "w") as f:
    json.dump(player_data, f, indent=2)

  return {"status": "success", "message": f"{template_name} recruited!", "player": player_data}
