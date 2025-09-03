# app/main.py
from fastapi import FastAPI
from app.services import battle

app = FastAPI(
    title="Creature Battle AI",
    description="FastAPI endpoints for simulating and training AI creatures",
    version="1.0"
)

# Include battle router
app.include_router(battle.router, prefix="/battle", tags=["Battle"])
