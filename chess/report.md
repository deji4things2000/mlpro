**Course:** Artificial Intelligence  
**Author:** Adedeji Sunday Adediran  
**Institution:** Dartmouth College  
**Date:** October 2025  

---

## 1. Description of Implementation

This project implements a **chess engine** with multiple AI agents, focusing on the core **adversarial search algorithms: Minimax and Alpha-Beta Pruning**. The system is modular, allowing different AI players (Human, Random, Minimax, AlphaBeta) to compete against each other via both a text-based and a graphical interface.

---

### a. How the Implemented Algorithms Work

#### **MinimaxAI (MinimaxAI.py)**
This AI models chess as a two-player, zero-sum game, exploring the game tree up to a specified depth.

- **Maximizing Player (White):** Finds moves leading to the highest evaluation score.  
- **Minimizing Player (Black):** Finds moves leading to the lowest evaluation score.  
- **Process:** The algorithm recursively evaluates all legal moves. At leaf nodes (set by depth limit or terminal game state), it uses a static evaluation function. Scores propagate upward as each player selects the best move from their perspective.

#### **AlphaBetaAI (AlphaBetaAI.py)**
An optimized version of Minimax that prunes unnecessary nodes.

- **Alpha (α):** Minimum score guaranteed to the maximizing player (starts at −∞).  
- **Beta (β):** Maximum score guaranteed to the minimizing player (starts at +∞).  
- **Pruning:** When a branch cannot influence the final decision (β ≤ α), the algorithm prunes it, skipping unnecessary evaluations.

#### **Iterative Deepening**
Both AIs include optional iterative deepening, incrementally searching from shallow to deeper depths. This allows **time-bounded search** and improves **move ordering**, as strong moves from previous depths are prioritized.

---

### b. Key Design Decisions

- **Unified AI Structure:** Both AIs share a common architecture, making them easily interchangeable in `ChessGame.py`.  
- **Evaluation Function:** Material-based (Pawn=1, Knight/Bishop=3, Rook=5, Queen=9). Checkmate = ±1000; draw = 0.  
- **Move Ordering:** AlphaBetaAI prioritizes captures and checks, significantly improving pruning efficiency.  
- **Node Counting:** Both AIs track `nodes_visited`, measuring efficiency and depth performance.  
- **Safety Mechanism:** If an AI returns an illegal move, `make_move()` defaults to the first legal move to prevent crashes.

---

## 2. Evaluation of the Implemented Algorithms

### Do the algorithms work?
Yes. Both **Minimax** and **Alpha-Beta** perform correctly, producing legal moves and displaying superior play compared to **RandomAI**.

### How well do they work?

- **Against RandomAI:** Both consistently win, capturing undefended pieces and executing basic checkmates.  
- **Playing Strength:** Increases with search depth. Depth 2–3 yields beginner-level performance; deeper searches are limited by the **horizon effect**.  
- **Partial Success:** The AIs avoid blunders, exploit mistakes, and use alpha-beta pruning effectively. Iterative deepening further enhances performance.  
- **Overall:** A complete success demonstrating practical application of adversarial search theory.

---

## 3. Responses to Discussion Questions

---

### 1) Minimax and Cutoff Test
**Observation:** Minimax performance degrades exponentially with depth due to chess’s high branching factor (~35–40 moves per position).

| Depth | Nodes Visited | Performance |
|-------|----------------|-------------|
| 1 | Dozens | Instant |
| 2 | Hundreds | Very fast |
| 3 | Thousands | Few seconds |
| 4 | Hundreds of thousands | Tens of seconds |

The **cutoff test** (`depth == 0` or terminal state) halts recursion effectively. This demonstrates the necessity of **alpha-beta pruning** for deeper searches.

---

### 2) Evaluation Function
The evaluation function is **purely material-based**, ignoring positional aspects like king safety or pawn structure.

#### **Varying Depth:**
- **Low Depth (1–2):** AI is short-sighted, prioritizing immediate gains and falling for traps.  
- **Higher Depth (3–4):** Becomes tactically aware, avoiding forks and recognizing multi-move tactics.  

**Insight:** Even with a simple evaluation, deeper search dramatically improves play strength.

---

### 3) Alpha-Beta
**Observation:** Move ordering has a major impact on performance.

- **Without Move Ordering:** Similar node count to Minimax; limited pruning.  
- **With Move Ordering:** Visits **5–20× fewer nodes**, establishing efficient α/β bounds early.  

**Result:** Move ordering is essential for maximizing alpha-beta efficiency.

---

### 4) Iterative Deepening
**Observation:** The best move may change as depth increases.

- **Consistency:** Often stable across depths (e.g., e2e4 remains optimal).  
- **Improvement Example:**
  - **Depth 2:** Chooses `Qxd5` for material gain.  
  - **Depth 3:** Rejects `Qxd5` after foreseeing `Nc6` counterattack, instead selecting `Nf3`.  

**Conclusion:** Iterative deepening enhances tactical foresight and ensures fallback decisions under time constraints.

---

**End of Report**
