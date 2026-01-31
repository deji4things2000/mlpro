# Transformer-Based Decision Agent in Gymnasium

Transformer policy trained via imitation learning for navigation in a custom Gymnasium environment with RGB + LIDAR + state observations.

## 🚀 Quick Start

### 1. Clone & Navigate
```bash
git clone <your-repo> transformer-agent
cd transformer-agent
# Option A: Conda (preferred for ML)
conda env create -f environment.yml
conda activate transformer-agent

# Option B: Manual conda + pip
conda create -n transformer-agent python=3.10
conda activate transformer-agent
pip install -r requirements.txt

python -c "from env.nav_env import NavEnv; env=NavEnv(); print('✅ Env OK:', env.observation_space)"

python train/collect_expert.py


python train/train_il.py

python eval/evaluate.py


To reproduce result exactly, run:

conda env create -f environment.yml && python train/train_il.py