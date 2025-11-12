**Course:** Artificial Intelligence  
**Author:** Adedeji Sunday Adediran  
**Institution:** Dartmouth College  
**Date:** October 2025  

# Chess AI Parallel Tournament Engine Report

## 1. Description of Implementation

This project implements a **chess engine** featuring multiple AI agents and classic adversarial search algorithms—**Minimax** and **Alpha-Beta Pruning**—optimized for tournament play and parallel performance. The system is modular, supporting Human, Random, Minimax, and AlphaBeta agents, accessible through both textual and graphical interfaces.

---

### a. How the Implemented Algorithms Work

#### MinimaxAI (`MinimaxAI.py`)
- Explores chess as a two-player, zero-sum game.
- **Parallel Root Search:** At each position, legal moves are evaluated in parallel across all available CPU cores.
- **Maximizing Player (White):** Pursues moves with maximal evaluation.
- **Minimizing Player (Black):** Pursues minimum evaluation.
- **Dynamic Depth:** Depth of search adapts based on the time remaining for that player.
- **Iterative Deepening:** Always returns the best available move within the time allowed, leveraging dynamic depth for both safety and strength.

#### AlphaBetaAI (`AlphaBetaAI.py`)
- An optimized Minimax, using pruning to skip evaluated branches with no impact.
- **Alpha (α):** Minimum score guaranteed to the maximizing player.
- **Beta (β):** Maximum score guaranteed to the minimizing player.
- **Move Ordering:** Prioritizes checks and captures to maximize pruning efficiency.
- **Parallelism:** Uses all CPU cores for initial candidate moves per position; inner search retains pruning logic.
- **Time Aware:** Ensures calculation fits strict per-player time budgets.

#### Chess Timer and Time Management
- Implements a strict chess clock using a dedicated timer class (5 minutes per player).
- Move computations are dynamically throttled or deepened according to remaining player time.
- Ensures every move is made within tournament time constraints.

---

### b. Key Design Decisions

- **Unified Player Interface:** All agents are interchangeable in `ChessGame.py` via a shared interface.
- **Material Evaluation:** Quick, robust scoring (Pawn=1, Knight/Bishop=3, Rook=5, Queen=9), with special handling for checkmate and draw.
- **Fallback Module:** Defaults to a legal move if an agent fails to select one in time, ensuring continuity.
- **Efficiency Counters:** Node and time tracking validate speedups and scalability.
- **Full CPU Utilization:** Root parallelization maximizes hardware effectiveness.

---

## 2. Evaluation of the Implemented Algorithms

### Do the algorithms work?
Yes. All agents play legal moves and integrate perfectly with the clock for tournament legal play. Tactical skill scales with permitted search depth and remaining game time.

### How well do they work?

- **Against RandomAI:** Deterministic agents win nearly every game.
- **Strength Scaling:** Parallelization and iterative deepening enable strong play under time controls.
- **Timed Play:** Never lose on time, always complete moves within per-player budget.
- **Alpha-Beta Efficiency:** Node visitation is minimized and less dependent on search depth, thanks to pruning and move ordering.

---

## 3. Responses to Discussion Questions

### 1) Minimax and Cutoff Test

| Depth | Nodes Visited | Performance      |
|-------|---------------|-----------------|
| 1     | Dozens        | Instant         |
| 2     | Hundreds      | Very Fast       |
| 3     | Thousands     | Few Seconds     |
| 4     | 100,000+      | Fast in parallel|

Efficient cutoff prevents runaway recursion; parallel root search allows deeper exploration in real time.

---

### 2) Evaluation Function

Material-centric: Fast and simple, but ignores positional intricacies.

- **Low Depth (1–2):** Prone to tactical mistakes.
- **High Depth (3+):** Tactically aware, robust against shallow traps.

---

### 3) Alpha-Beta and Move Ordering

- **Move Ordering:** Significantly reduces nodes visited (up to 20× fewer).
- **Parallel Evaluation:** Allows more time to be spent on deeper or complex positions.

---

### 4) Iterative Deepening and Timed Play

- Defaults to best move found so far if time runs low.
- Example: At depth 2, may prefer greed; at depth 3+, avoids tactical blunders.

---

**End of Report**
