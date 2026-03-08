# ♟️ Reinforcement Learning Chess Agent

> An AI agent that teaches itself chess **from scratch** — no hardcoded rules, no opening books. It starts by making random moves and gradually improves by playing against itself, guided only by a reward signal.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation](#2-motivation)
3. [Architecture](#3-architecture)
4. [Training Pipeline](#4-training-pipeline)
5. [Results](#5-results)
6. [How to Run](#6-how-to-run)
7. [Demo](#7-demo)

---

## 1. Project Overview

This project builds a chess-playing AI using **Reinforcement Learning (RL)** — a technique where an agent learns by trial and error, earning rewards for good actions and penalties for bad ones. The agent starts with zero chess knowledge and improves purely by playing thousands of games against itself.

No chess strategy is hardcoded. The AI discovers concepts like captures, piece safety, and checkmate entirely on its own, through experience.

**What's included:**

- A **chess game environment** the agent can play in and learn from
- A **neural network** (the AI's "brain") that decides which move to make in any position
- A **reward system** that gives the AI useful feedback, especially in the early stages of training
- **Evaluation against Stockfish** — the world's strongest open-source chess engine — to measure how much the AI has actually improved
- An **interactive web interface** so anyone can sit down and play against the trained agent

---

## 2. Motivation

Chess is a fascinating problem for AI. It's a game that humans have studied for centuries, yet it's complex enough that no computer can simply look up the right answer — it has to think.

Here's why chess makes a great RL challenge:

| Property | Why it's hard |
|---|---|
| ~35 legal moves per turn | The AI has a lot of options to consider at every step |
| ~40 moves per game on average | A mistake early on might not cost you until much later |
| ~10⁴³ possible board positions | Memorising positions is impossible — the AI must learn to reason |
| No feedback until the game ends | The AI plays an entire game before finding out if it won or lost |

Despite this difficulty, chess is something humans universally understand. That makes it easy to see — and show others — whether the AI is genuinely improving. ELO ratings provide a precise, widely recognised way to measure that progress.

The goal was simple: build an agent that starts knowing nothing about chess, and watch it get meaningfully better over time.

---

## 3. Architecture

The system has four main components. Here's how they fit together:

---

### 3.1 How the Board is Represented

Before the AI can make a decision, it needs to "see" the board. Since neural networks work with numbers, the board is converted into a **stack of 12 grids**, each 8×8 squares:

```
Grids 0–5   →  White pieces: Pawn, Knight, Bishop, Rook, Queen, King
Grids 6–11  →  Black pieces: Pawn, Knight, Bishop, Rook, Queen, King
```

Each grid is filled with `1` where that piece sits on the board, and `0` everywhere else. The result is a compact, structured snapshot the neural network can process efficiently.

*File: `src/environment/board_encoder.py`*

---

### 3.2 The Policy Network — How the AI Picks a Move

This is the AI's decision-making brain. It takes the board snapshot as input and outputs a **ranked list of every possible move**, with a probability score for each. The AI picks from this list — favouring high-probability moves, but with some randomness to encourage exploration early in training.

```
Board snapshot (12 grids of 8×8)
         ↓
  Convolutional layers      ← detect patterns across the board
         ↓
  Fully connected layers    ← combine patterns into a decision
         ↓
  Move probability scores   ← one score per possible move
         ↓
  Illegal moves filtered out → AI picks from legal moves only
```

*File: `src/models/policy_network.py`*

---

### 3.3 The Value Network — Judging a Position

Alongside picking moves, the AI also learns to **estimate how good any board position is** — essentially asking "am I winning or losing right now?" This helps the AI plan ahead and makes training more stable overall.

*File: `src/models/value_network.py`*

---

### 3.4 The Reward System — Teaching Right from Wrong

The most natural reward is simple: win = +1, lose = −1. But with that alone, the AI gets almost no useful signal during the thousands of moves it plays before its first win.

To speed up early learning, small rewards are given for meaningful events during the game:

| Event | Reward | Why |
|---|---|---|
| Illegal move attempt | −0.5 | Teaches the AI to stay within the rules |
| Any legal move | +0.01 | Small encouragement to keep playing |
| Capture a pawn | +0.05 | Capturing material is generally good |
| Capture a piece | +0.10 | Bigger captures are more valuable |
| Lose a piece | −0.10 | Losing material is bad |
| Checkmate the opponent | +1.00 | The ultimate goal |
| Draw | 0.00 | Neutral outcome |

These rewards are kept small and balanced so the AI isn't tricked into, say, capturing pieces recklessly at the expense of its overall position.

*File: `src/environment/reward_system.py`*

---

## 4. Training Pipeline

Training runs in a continuous loop of three phases:

```
Start: neural network initialised with random weights (knows nothing)
│
└─ Repeat thousands of times:
      │
      ├─ 1. SELF-PLAY
      │    The agent plays multiple games against itself simultaneously
      │    Every move, board state, and reward is saved
      │
      ├─ 2. LEARNING
      │    Saved experience is used to update the neural network
      │    Moves that led to wins are reinforced
      │    Moves that led to losses are discouraged
      │
      └─ 3. EVALUATION
           The agent plays 10 games vs. Stockfish (beginner difficulty)
           Results are logged: win rate, average reward, game length
```

**Key files:**

| File | What it does |
|---|---|
| `src/training/self_play.py` | Runs self-play games to generate training data |
| `src/training/trainer.py` | Updates the neural network from recorded experience |
| `src/training/replay_buffer.py` | Stores and retrieves past experience efficiently |
| `src/environment/chess_env.py` | The chess environment — handles moves, rules, and rewards |
| `configs/training_config.yaml` | All training settings in one easy-to-edit file |

**How the agent develops over time:**

| Games Played | What the Agent Does |
|---|---|
| 0 – 1,000 | Makes random legal moves |
| ~5,000 | Starts making captures |
| ~20,000 | Stops making obvious blunders |
| ~50,000 | Shows signs of basic strategy |

Training takes roughly **several hours depending on hardware and configuration**, depending on settings.

---

## 5. Results

> ⚠️ **Training in progress.** This section will be updated with real results once training is complete.

**Metrics being tracked:**

- Average reward earned per game over training time
- Win rate against Stockfish (set to ELO 800 — beginner level)
- Average game length — longer, more structured games suggest better play
- Estimated ELO rating of the fully trained agent

**Charts to be added:**

| Chart | What it shows |
|---|---|
| `training_reward.png` | How reward improves as training progresses |
| `win_rate_vs_stockfish.png` | Win/draw/loss record against the engine over time |
| `game_length.png` | How game length changes as the agent improves |
| `elo_progression.png` | The agent's estimated ELO at each checkpoint |

---

## 6. How to Run

### What you'll need

- Python 3.10 or later
- The Stockfish chess engine installed on your machine ([download here](https://stockfishchess.org/download/))

### Setup

```bash
# 1. Download the project
git clone https://github.com/gurusainathp/chess-rl-agent.git
cd chess-rl-agent

# 2. Install all required libraries
pip install -r requirements.txt
```

### Train the agent

```bash
python scripts/train.py --config configs/training_config.yaml
```

The agent saves a checkpoint after each evaluation round to `data/models/`.

### Test the agent against Stockfish

```bash
python scripts/evaluate.py --model data/models/latest.pt --elo 800
```

### Play against the agent (terminal)

```bash
python scripts/play.py
```

### Play against the agent (web interface)

```bash
streamlit run src/ui/play_vs_ai.py
```

This opens a browser window with a fully interactive chessboard.

---

## 7. Demo

> 🖼️ **Screenshots coming soon** — interface and evaluation demos will be added once training is complete.

### Web Interface

The browser-based interface lets you:
- Play a full game against the trained agent — no extra setup beyond installation
- See the AI's evaluation score for the current board position
- Step back through the move history
- Restart the game or switch sides at any time

### What you see during training

While training runs, a live progress bar keeps you updated in the terminal:

```
[Iter 042 | Games: 4,200] Avg Reward: 0.134 | Win vs SF800: 12% | Avg Length: 47 moves
```

---

## 📁 Project Structure

```
chess-rl-agent/
├── README.md                     ← You are here
├── requirements.txt              ← All dependencies
├── configs/
│   └── training_config.yaml     ← Training settings (edit to experiment)
├── data/
│   ├── models/                   ← Saved model checkpoints
│   └── training_games/           ← Game records from self-play
├── notebooks/
│   ├── experiment_notes.ipynb    ← Research notes and findings
│   └── reward_experiments.ipynb  ← Testing different reward setups
├── scripts/
│   ├── train.py                  ← Start training
│   ├── evaluate.py               ← Test against Stockfish
│   └── play.py                   ← Play against the agent
└── src/
    ├── environment/
    │   ├── chess_env.py          ← Chess game logic and rules
    │   ├── board_encoder.py      ← Converts board to numbers for the network
    │   └── reward_system.py      ← Defines rewards and penalties
    ├── models/
    │   ├── policy_network.py     ← Move selection network
    │   └── value_network.py      ← Position evaluation network
    ├── training/
    │   ├── self_play.py          ← Generates games for training data
    │   ├── trainer.py            ← Updates the network from experience
    │   └── replay_buffer.py      ← Stores and retrieves past experience
    ├── evaluation/
    │   ├── elo_evaluator.py      ← Estimates agent ELO rating
    │   └── play_vs_engine.py     ← Runs games against Stockfish
    ├── ui/
    │   └── play_vs_ai.py         ← Streamlit web interface
    └── utils/
        ├── move_encoder.py       ← Converts moves to numbers and back
        └── logging_utils.py      ← Training logs and metrics
```

---

## 🛠️ What This Project Demonstrates

| Area | Details |
|---|---|
| Reinforcement Learning | Self-play training loop, reward shaping, policy optimisation |
| Neural Networks | Convolutional architecture, move masking, value estimation |
| ML Engineering | Experience replay, model checkpointing, config-driven experiments |
| Game Environment Design | Custom environment with legal move validation and reward emission |
| Model Evaluation | ELO benchmarking against a real chess engine |
| Software Engineering | Clean modular codebase, separation of concerns, readable structure |
| User Interface | Interactive browser-based play via Streamlit |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.