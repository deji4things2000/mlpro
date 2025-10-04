# Chess AI - Adversarial Search Implementation

A Python chess implementation featuring multiple AI agents using adversarial search algorithms including Minimax, Alpha-Beta Pruning, and Iterative Deepening.

## 🎯 Project Overview

This project implements chess-playing AI agents as part of a study in adversarial search algorithms. The system includes multiple AI implementations with varying levels of sophistication, from random move selection to advanced game tree search with pruning.

## 🚀 Quick Start

### Prerequisites
- Python 3.6 or higher
- `python-chess` library

### Installation
1. **Clone or download** all project files to a directory
2. **Install dependencies**:
   ```bash
   pip install python-chess

### Running the Game

1. Text Interface: python test_chess.py
2. Graphical Interface: 
    pip install PyQt5
    python gui_chess.py

### Project Structure

chess-ai/
├── test_chess.py          # Main text interface
├── gui_chess.py           # Graphical interface (optional)
├── ChessGame.py           # Game state management
├── HumanPlayer.py         # Human input handler
├── RandomAI.py            # Random move AI
├── MinimaxAI.py           # Minimax algorithm
├── AlphaBetaAI.py         # Alpha-Beta algorithm
└── README.md              # This file

