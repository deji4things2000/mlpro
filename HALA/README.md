# HALA: Hybrid Approximate Linear Algebra for Robot Dynamics

HALA explores fast approximate solvers for robot dynamics M(q) q̈ + C(q, q̇) + g(q) = τ using Pinocchio. It benchmarks:
- Gauss-Jordan (numpy.linalg.solve baseline)
- Neumann series preconditioning
- SPAI (diagonal sparse approximate inverse)
- HALA: SPAI + truncated Neumann refinement with stability fallback

## Requirements
- Python 3.9+
- Pinocchio
- NumPy, SciPy
- Optional: Jupyter

## Install (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pinocchio numpy scipy
```

## Structure
- pinocchio.ipynb — setup, model loading, benchmarking
- robot_arm_2link.urdf / ur5robot.urdf — sample URDFs

## Usage
Run HALA/pinocchio.ipynb:
```python
import pinocchio as pin, numpy as np
model = pin.buildModelFromUrdf("/Users/user_1/mlpro/HALA/ur5robot.urdf")
data = model.createData()
# computeAllTerms -> get M, C(q, q̇), g, then solve
qddot_ref = np.linalg.solve(data.M, b)
qddot_hala = hala(data.M, b, num_neumann=10)
```

## HALA (simplified)
```python
def spai_inverse(M):
    return np.diag(1 / np.diag(M))

def hala(M, b, num_neumann=10):
    G = spai_inverse(M)
    E = np.eye(M.shape[0]) - G @ M
    S = np.eye(M.shape[0]); term = np.eye(M.shape[0])
    for _ in range(1, num_neumann):
        term = term @ E; S = S + term
    x = (S @ G) @ b
    if np.linalg.norm(M @ x - b) > 1e-3:
        x = np.linalg.solve(M, b)
    return x
```

## Tips
- Clamp non-finite/extreme velocity limits before sampling:
```python
qd_limit_safe = np.copy(model.velocityLimit)
qd_limit_safe[~np.isfinite(qd_limit_safe)] = 1.0
qd_limit_safe = np.clip(qd_limit_safe, 0, 10)
```
- Use small perturbations (M + δM) and kappa to assess stability.
- Adjust URDF paths for your environment.

## Results to report
- Avg time (ms)
- Relative error vs baseline
- Stability (kappa)