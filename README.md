# MLPRO

A collection of AI/ML and optimization projects, algorithms, and experiments. This repo includes metaheuristics, robotics dynamics, search/constraint solvers, and educational assignments.

## Project Index

- HALA/
  - Hybrid Approximate Linear Algebra for robot dynamics with Pinocchio. Benchmarks Gauss-Jordan, Neumann, SPAI, and the HALA hybrid.
  - See HALA/README.md.

- foxes_and_chicken_search_algorithm/
  - Predator–prey metaheuristic (FCSA). Foxes exploit, chickens explore. Continuous optimization with pluggable objectives.
  - See foxes_and_chicken_search_algorithm/README.md.

- mazeworld/
  - Grid/maze search experiments (e.g., BFS/DFS/A*, heuristics). Likely includes agents and path planning.

- CSP/
  - Constraint Satisfaction Problems: backtracking, arc consistency, heuristics (MRV, degree, LCV), and problem examples.

- chess/
  - Chess engine utilities or algorithms (move generation, evaluation, search). May include unit tests and experiments.

- chess_tournament/
  - Tournament scheduling or simulation for chess (pairings, scoring, standings). Useful for combinatorial optimization demos.

- Logic_PA5/
  - Logic programming/assignment 5 resources (parsers, evaluators, or proofs). Educational or course-related.

- Others/
  - Misc experiments and utilities.
  - optimizing_mat_mul_and_cuda/: matrix multiplication optimization and CUDA notes/code.
  - .vscode/: workspace settings.

- farm_livestock_portal/
  - Desktop portal for farm livestock tracking (Tkinter + MySQL) with barcode/QR workflows and AI-assisted health inference.
  - See [farm_livestock_portal/README.md](farm_livestock_portal/README.md).

## Getting Started (macOS)

```bash
# Clone and enter (if not already)
git clone <your-repo-url> mlpro
cd mlpro

# Python environment
python3 -m venv .venv
source .venv/bin/activate

# Install common deps (adjust per project)
pip install numpy scipy jupyter

# Optional: open in VS Code
open -a "Visual Studio Code" .
```

## Running

- HALA: open HALA/pinocchio.ipynb and run cells; see HALA/README.md.
- FCSA: see foxes_and_chicken_search_algorithm/README.md; run examples or tests.
- CSP/Mazeworld/Chess: open modules or notebooks in each folder and run via Python or Jupyter.

### Farm Livestock Portal

- Overview: GUI to register livestock, generate and preview barcodes/QR codes, manage health records, and auto-update `health_status` based on health entries. Species/breeds are sourced from FAO CSV.
- Quick setup (uses its own requirements):

```bash
cd farm_livestock_portal
pip install -r requirements.txt
```

- Run the app (from the portal folder):

```bash
python -m gui.main_app
```

- Seed dummy data (N animals + one “sick bison” health record):

```bash
python -m scripts.seed --animals 100
```

- Optional ML model for health inference (fallback heuristic used if not present):

```bash
python -c "from services.health_ml import train_health_model; train_health_model()"
```

The trained model is saved to [farm_livestock_portal/models/health_model.pkl](farm_livestock_portal/models/health_model.pkl) and is picked up automatically when available.

## Tests

```bash
# If projects use pytest:
pytest -q
```

## Contributing

- Keep each project self-contained with its own README and requirements.txt.
- Use consistent code style, small functions, and unit tests.
- Add brief benchmarks for algorithmic projects.

## License

MIT. Individual subprojects may have their own licenses.
