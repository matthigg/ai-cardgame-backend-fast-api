# AI Battle Simulator Backend

This is a **FastAPI backend** for a turn-based AI battle simulation between two creatures. Each creature uses a neural network to make decisions in real-time battles. The backend supports generating battle logs, persisting neural network checkpoints, and customizing creature stats and abilities.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Notes](#notes)

---

## Features

- Turn-based AI battles between configurable creatures.
- Reinforcement learning using REINFORCE policy gradient.
- Neural network persistence via checkpoints.
- Configurable special abilities, rewards, and stats per creature.
- Generates detailed battle logs and summary statistics.
- Ready for integration with a frontend (Angular, React, etc.).

---

## Requirements

- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/installation/)
- Optional: [virtualenv](https://docs.python.org/3/library/venv.html) for isolated environments.

---

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/matthigg/fast-api-ai-cardgame.git
cd fast-api-ai-cardgame
```

2. **Create and activate a virtual environment (recommended)**

```bash
python -m venv venv
```

## Git Bash

```bash
source venv/Scripts/activate
```

## Windows

```bash
venv\Scripts\activate
```

## macOS/Linux

```bash
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Verify installation**

```bash
python -m pip list
```

Ensure fastapi, uvicorn, torch, numpy, and other dependencies are installed.

## Running the App

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

--reload automatically reloads the server on code changes.

Default URL: http://127.0.0.1:8000

## API Documentation 

FastAPI automatically provides interactive API documentation:

Swagger UI: http://127.0.0.1:8000/docs

ReDoc: http://127.0.0.1:8000/redoc

You can test all endpoints directly from the Swagger UI, including simulating battles and retrieving logs.

## Project Structure

```
fast-api-ai-cardgame/
│
├─ app/
│  ├─ main.py                   # FastAPI entrypoint
│  ├─ config.py                 # Creature stats, rewards, and global configs
│  ├─ modules/
│  │  ├─ creature.py            # Creature class & initialization
│  │  ├─ battle_simulation.py   # Battle loop & tick logic
│  │  ├─ neural_network.py      # Neural network & REINFORCE update
│  │  ├─ network_persistence.py # Checkpoint save/load
│  │  ├─ logging_utils.py       # Log helpers
│  │  ├─ utils.py               # Misc utilities
│  │  └─ training_loop.py       # Orchestrates epochs and battle simulations
│  │
│  └─ services/
│     └─ battle.py              # Battle service layer (FastAPI endpoints)
│
├─ checkpoints/                 # Saved neural network checkpoints
├─ battle_logs/                 # Generated battle logs
├─ requirements.txt
└─ README.md

```

## Notes:

Creature Configs: Modify CREATURES in config.py to adjust HP, energy, speed, special abilities, and reward parameters.

Checkpoints: If you add or remove special abilities, the corresponding checkpoint files should be automatically deleted to avoid inconsistencies.

Seed: By default, the simulation uses a fixed random seed for reproducibility. Disable it in config.py if you want varied outcomes.

Batch Size: Configure epoch_batch_size and log_batch_size in config.py to control how many battles are simulated per run and logging frequency.
