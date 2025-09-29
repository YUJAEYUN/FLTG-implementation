# FLTG Implementation & Validation

## Quick Start

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd FLTG
```

### 2. Install dependencies
```bash
pip install torch torchvision matplotlib numpy scipy
```

### 3. Run experiments

#### Baseline (No attack)
```bash
cd FL-Byzantine-Library
python3 main.py --dataset_name mnist --nn_name mnistnet --num_client 20 --traitor 0 --aggr avg --trials 1 --global_epoch 10 --gpu_id -1 --bs 64
```

#### ROP Attack with different defenses
```bash
# FedAVG (no defense)
python3 main.py --dataset_name mnist --nn_name mnistnet --num_client 20 --traitor 0.2 --attack rop --aggr avg --trials 1 --global_epoch 10 --gpu_id -1 --bs 64

# Krum
python3 main.py --dataset_name mnist --nn_name mnistnet --num_client 20 --traitor 0.2 --attack rop --aggr krum --trials 1 --global_epoch 10 --gpu_id -1 --bs 64

# Trimmed-Mean
python3 main.py --dataset_name mnist --nn_name mnistnet --num_client 20 --traitor 0.2 --attack rop --aggr tm --trials 1 --global_epoch 10 --gpu_id -1 --bs 64

# FLTG (our implementation)
python3 main.py --dataset_name mnist --nn_name mnistnet --num_client 20 --traitor 0.2 --attack rop --aggr fltg --trials 1 --global_epoch 10 --gpu_id -1 --bs 64
```

## Project Structure

```
FLTG/
├── FL-Byzantine-Library/        # Base framework
│   ├── Aggregators/
│   │   ├── fltg.py             # FLTG implementation (NEW)
│   │   └── ...                 # Other aggregators
│   ├── Attacks/                # Byzantine attacks
│   ├── Models/                 # Neural network models
│   ├── main.py                 # Main entry point
│   └── mapper.py               # Aggregator/Attack mapper (MODIFIED)
└── README_SETUP.txt            # This file

```

## Key Files Modified/Added

1. **FL-Byzantine-Library/Aggregators/fltg.py** - NEW
   - FLTG aggregator implementation
   - ReLU-clipped cosine similarity filtering
   - Dynamic reference selection
   - Non-IID aware weighting
   - Magnitude normalization

2. **FL-Byzantine-Library/mapper.py** - MODIFIED
   - Added FLTG to aggregator mapper
   - Added root dataset initialization for FLTG

3. **FL-Byzantine-Library/parameters.py** - MODIFIED
   - Added missing buck_rand parameter

4. **FL-Byzantine-Library/main.py** - MODIFIED
   - Fixed save_results call signature

## Experiment Parameters

- Dataset: MNIST
- Model: MNISTNET (0.431M parameters)
- Clients: 20
- Byzantine ratio: 20% (4 malicious clients)
- Attack: ROP (Relocated Orthogonal Perturbation)
- Epochs: 10
- Batch size: 64

## Dependencies

- Python 3.7+
- PyTorch
- torchvision
- numpy
- scipy
- matplotlib

## Notes

- First run will download MNIST dataset automatically
- GPU is optional (use --gpu_id 0 for GPU, --gpu_id -1 for CPU)
- Results are saved automatically with timestamp