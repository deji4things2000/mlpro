# CNN, BPE, and Decoder-Only Transformer Experiments

This repository contains three distinct implementations, each in its own Python module / notebook, along with everything needed to run the experiments from scratch and reproduce the reported results (fixed random seeds and a Python environment specification).

## 1. Project Overview

The repository is organized around three core components:

- **CNN Representational Power**  
  An implementation and experiment studying the representational power of convolutional neural networks.

- **Byte Pair Encoding (BPE) Algorithm**  
  A from-scratch implementation of the BPE tokenization algorithm trained on a Shakespeare corpus (`shakespeare.txt`).

- **Decoder-Only Transformer**  
  A decoder-only Transformer model (GPT-style) trained as a character-level language model on the same corpus. 

Each component is implemented in its own Python file or notebook and can be run independently.
## 2. Reproducibility

To ensure experiments are fully reproducible:

- All main scripts fix random seeds for Python, NumPy, and (where applicable) PyTorch.  
- A `requirements.txt` file specifies the exact Python package versions used.  
- Data preprocessing, training, and evaluation are implemented as scripted steps that can be rerun end-to-end.

### 2.1 Fixed Random Seeds

Each main Python file begins with a seeding block of the form:

import random
import numpy as np

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
For PyTorch-based experiments

import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


This ensures that multiple runs with the same environment and inputs produce identical results.

### 2.2 Python Environment / Requirements

All dependencies are specified in `requirements.txt`, for example:


torch==2.2.0
numpy==1.26.0
tqdm==4.66.0
jupyter==1.0.0
matplotlib==3.8.0


To create an isolated environment and install the dependencies:

python -m venv .venv
Linux / macOS

source .venv/bin/activate
Windows (PowerShell)
.venv\Scripts\Activate.ps1


## 3. Repository Structure

A typical layout for this repository is:

├── README.md
├── data/
│ └── shakespeare.txt
├── cnn_representational_power.py # or CNN_Representational_Power.ipynb
├── byte_pair_encoding_algorithm.py # or Byte_Pair_Encoding_Algorithm.ipynb
├── decoder_only_transformer.py # or Decoder_Only_Transformer.ipynb
└── transformer/


- `data/shakespeare.txt`: corpus used for BPE and Transformer experiments. Update the file path in the python file.
- `cnn_representational_power.py` / `CNN_Representational_Power.ipynb`: CNN representational power experiments.   
- `byte_pair_encoding_algorithm.py` / `Byte_Pair_Encoding_Algorithm.ipynb`: BPE learning and encoding logic.  
- `decoder_only_transformer.py` / `Decoder_Only_Transformer.ipynb`: decoder-only Transformer model and training code. 

## 4. Running Each Implementation

### 4.1 CNN Representational Power

To run the CNN experiment from the Python file:

python cnn_representational_power.py


or, if using the notebook:

1. Launch Jupyter: `jupyter notebook`  
2. Open `CNN_Representational_Power.ipynb`  
3. Run all cells in order.

The script/notebook reports the experimental setup and metrics for the CNN architecture under study. 

### 4.2 Byte Pair Encoding (BPE) Algorithm

Ensure `data/shakespeare.txt` exists, then run:

python byte_pair_encoding_algorithm.py
--data_path data/shakespeare.txt
--num_merges 1000
--output_dir artifacts/bpe/



This implementation:

- Initializes the vocabulary as all unique characters in the corpus.  
- Repeatedly finds the most frequent adjacent pair of tokens, merges it into a new token, updates the vocabulary, and retokenizes the corpus.  
- Saves the learned merge rules and final vocabulary to `artifacts/bpe/`.

If using the notebook, open `Byte_Pair_Encoding_Algorithm.ipynb` and run all cells. 

### 4.3 Decoder-Only Transformer

With `data/shakespeare.txt` in place, run:

python decoder_only_transformer.py
--data_path data/shakespeare.txt
--block_size 128
--d_model 128
--n_heads 4
--n_layers 2
--batch_size 32
--max_iters 3000
--seed 1337
--output_dir artifacts/transformer/



This script:

- Builds a character-level vocabulary from the corpus.  
- Trains a decoder-only Transformer using teacher forcing and cross-entropy loss.  
- Saves the trained model and logs to `artifacts/transformer/`.

To generate sample text (e.g., 10–15 lines with a prompt such as `JULIET:`):

python decoder_only_transformer.py
--data_path data/shakespeare.txt
--load_from artifacts/transformer/model.pt
--generate
--prompt "JULIET:"
--max_new_tokens 400



This produces qualitative samples matching the generation shown in the report. 

## 5. Notes for Reviewers

- Each Python file corresponds to a **distinct implementation** (CNN, BPE, Transformer) and can be run independently to reproduce the associated results.
- The combination of fixed random seeds, `requirements.txt`, and explicit run commands in this README ensures that all experiments are fully reproducible from scratch.