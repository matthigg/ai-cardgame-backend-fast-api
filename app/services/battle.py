import json
import os
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.config import CONFIG
from app.modules.training_loop import training_loop
from app.modules.training_loop_stream import training_loop_stream

router = APIRouter()

@router.get("/train")
def train_endpoint():
  training_loop()
  return {"status": "Training completed"}

@router.get("/training-stream")
async def training_stream():
  return StreamingResponse(training_loop_stream(), media_type="text/event-stream")

@router.get("/summary")
def get_summary():
  filename = os.path.join(CONFIG['log_dir'], 'summary.json')
  if os.path.exists(filename):
    with open(filename, 'r') as f:
      return json.load(f)
  return {"error": "Summary not available yet"}
