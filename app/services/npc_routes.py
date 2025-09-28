import os
import glob
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.modules.player_manager import create_npc
from app.config import CREATURE_TEMPLATES, GENERATED_DIR, NPCS_DIR

router = APIRouter(tags=["npc"])

class CreateNpcRequest(BaseModel):
  name: str
  creature: str

@router.post("/npc/create")
def create_new_npc(req: CreateNpcRequest):
  name = req.name.strip()
  creature = req.creature.strip()

  if not name:
    raise HTTPException(status_code=400, detail="NPC name cannot be empty")
  if creature not in CREATURE_TEMPLATES:
    raise HTTPException(
      status_code=400,
      detail=f"Invalid creature '{creature}'."
    )

  npc, path = create_npc(name, [creature])
  with open(path, "r") as f:
    return json.load(f)

# ----------  4) Get All NPC Data ----------
@router.get("/npcs")
def get_all_npcs():
  """
  Returns a list of all NPC JSON objects found in the NPC directory.
  """
  npc_dir = os.path.join(GENERATED_DIR, NPCS_DIR)  # adjust if your NPCs live elsewhere

  if not os.path.isdir(npc_dir):
    raise HTTPException(status_code=404, detail="NPC directory not found")

  npc_files = glob.glob(os.path.join(npc_dir, "*.json"))
  if not npc_files:
    raise HTTPException(status_code=404, detail="No NPC files found")

  npcs = []
  for path in npc_files:
    try:
      with open(path, "r") as f:
        npcs.append(json.load(f))
    except json.JSONDecodeError:
      # Skip bad files but continue loading others
      continue

  return npcs
