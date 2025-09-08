from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services import battle_routes
import os

app = FastAPI()

# ✅ Explicitly list allowed origins
# For dev, just your Angular app
# For prod, replace with your deployed frontend's URL (e.g. "https://yourapp.com")
origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
]

# Optionally: read from env var in prod for flexibility
# e.g., ORIGINS="https://yourapp.com,http://localhost:4200"
if "ORIGINS" in os.environ:
    origins = os.environ["ORIGINS"].split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # ✅ Safer than "*"
    allow_credentials=True,        # Needed if using cookies / auth headers
    allow_methods=["GET", "POST"], # Limit methods you actually use
    allow_headers=["Content-Type", "Authorization"], # Limit headers
)

# Include your routes
app.include_router(battle_routes.router, prefix="/battle", tags=["Battle"])
