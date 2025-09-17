from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services import battle_routes
from app.services import player_routes   # 👈 import your player routes
import os

app = FastAPI()

# ✅ Explicitly list allowed origins
origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
]

if "ORIGINS" in os.environ:
    origins = os.environ["ORIGINS"].split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Include your routes
app.include_router(battle_routes.router, prefix="/battle", tags=["Battle"])
app.include_router(player_routes.router, prefix="/player", tags=["Player"])  # 👈 add this

# Bootstrap
from app.modules.startup import bootstrap_players

bootstrap_players()
