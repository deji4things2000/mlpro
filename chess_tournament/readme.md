**Course:** Artificial Intelligence  
**Author:** Adedeji Sunday Adediran  
**Institution:** Dartmouth College  
**Date:** October 2025 

# Chess AI Parallel Engine

A modular Python chess engine with multiple AI agents (Human, Random, Minimax, Alpha-Beta), advanced parallelization, and strict chess clock management—designed for both casual and tournament competition.

---

## Features

- **Multiple Agents:** Human, Random, MinimaxAI, and AlphaBetaAI.
- **Parallel Search:** Uses all CPU cores at the root of search for fast, deep computation.
- **Alpha-Beta Pruning:** Efficiently skips unnecessary move branches for strong, rapid play.
- **Iterative Deepening:** Always returns best move found so far within time limits.
- **Tournament Timer:** Implements a 5-minute chess clock for each player.
- **Material Evaluation:** Simple, robust static evaluation using chess piece values.
- **CLI and GUI:** Play in terminal or with a graphical interface (Qt, optional).

---

## Quick Start

### Prerequisites

- Python 3.6+
- [python-chess](https://python-chess.readthedocs.io/)
- (Optional) PyQt5 for GUI

Install requirements: 
-pip install python-chess
-pip install pyQT5 for GUI



### Running the Game

- **Text Interface:**

    ```
    python testchess.py
    ```

- **Graphical Interface (Recommended for AI Tournaments):**

    ```
    python gui_chess.py
    ```

---

## Project Structure

- `ChessGame.py` – Game state manager
- `HumanPlayer.py` – Human move handler (CLI)
- `RandomAI.py` – Random-move agent (baseline)
- `MinimaxAI.py` – Parallel, time-aware Minimax AI
- `AlphaBetaAI.py` – Parallel, time-aware Alpha-Beta AI
- `gui_chess.py` – PyQt-based graphical interface
- `readme.md` – This file
- `report.md` – In-depth project and algorithmic report

---

## How It Works

- **Custom Matchups:** Select any combination of agents for White and Black.
- **Time Control:** Each player (AI or human) starts with 5 minutes for all moves—just like tournament chess.
- **Automatic Depth Tuning:** The AI chooses deeper searches when time is plentiful and responds rapidly as the clock runs low.
- **CPU Scaling:** The engine detects your hardware and uses all available cores for rapid analysis.

---

## Example Usage

Upon running, you can set up matches like:

- Human vs AI
- AlphaBetaAI vs MinimaxAI
- RandomAI vs AlphaBetaAI

---

## Limitations and Notes

- **Evaluation**: Currently material-only (no positional or endgame heuristics).
- **GUI**: Requires PyQt5. See comments in `gui_chess.py` for install instructions.
- **Legal Moves**: If the AI ever fails to make a move in time, the game engine automatically plays the first legal move.

---

## Credits

- Built using [python-chess](https://python-chess.readthedocs.io/).
- Heavily inspired by classic adversarial search literature.
- Developed for parallel, competitive AI research and demonstration.

---

## License

MIT License

---


