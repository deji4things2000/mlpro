# Foxes and Chicken Search Algorithm (FCSA)

A metaheuristic inspired by predator–prey dynamics. Foxes (exploiters) pursue chickens (explorers), balancing global exploration with local exploitation to optimize continuous objective functions.

## Features
- Chicken flock exploration with stochastic movement
- Fox pursuit and local refinement
- Bounds/constraints support
- Pluggable objectives and callbacks

## Quickstart
```python
from fcsa import FCSA, Sphere  # adjust import to your package layout

algo = FCSA(
    population_size=50,
    dims=30,
    bounds=[(-5.12, 5.12)] * 30,
    max_iters=1000,
    seed=42,
)

best_x, best_f = algo.optimize(Sphere())
print("Best score:", best_f)
```

## API (example)
- FCSA(population_size, dims, bounds, max_iters, seed=None, alpha=0.7, beta=0.3, inertia=0.9)
- optimize(objective, callback=None) -> (x_best, f_best)
- Objective: callable f(x: np.ndarray) -> float

## Tips
- Increase population_size for multimodal functions
- Use adaptive alpha/beta to shift exploration→exploitation
- Apply penalties for constraint violations

## Benchmarks
Report results for Sphere, Rastrigin, Rosenbrock:
- Best/mean score, evaluations, runtime
- Parameter settings used

## Development
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## License
MIT 