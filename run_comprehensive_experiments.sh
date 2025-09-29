#!/bin/bash

# FLTG Comprehensive Experiments
# This script runs extensive experiments to properly validate FLTG performance
# Following paper's methodology: multiple datasets, attack types, and Byzantine ratios

cd FL-Byzantine-Library

echo "=================================="
echo "FLTG Comprehensive Experiments"
echo "=================================="
echo ""

# Create results directory
mkdir -p ../results
RESULTS_DIR="../results"

# Common parameters
DATASET="mnist"
MODEL="mnistnet"
CLIENTS=20
EPOCHS=50  # Increased from 10 to 50
BS=64
GPU=-1

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Model: $MODEL"
echo "  Clients: $CLIENTS"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BS"
echo ""

# ==================================================
# Experiment 1: Baseline (No Attack) - Extended
# ==================================================
echo "=== Experiment 1: Baseline (No Attack, 50 epochs) ==="
python3 main.py \
  --dataset_name $DATASET --nn_name $MODEL \
  --num_client $CLIENTS --traitor 0 \
  --aggr avg --trials 1 --global_epoch $EPOCHS \
  --gpu_id $GPU --bs $BS \
  2>&1 | tee "$RESULTS_DIR/exp1_baseline_50ep.log"
echo "✓ Completed"
echo ""

# ==================================================
# Experiment 2: Byzantine Ratio Scaling (ROP attack)
# ==================================================
echo "=== Experiment 2: Byzantine Ratio Scaling ==="

for RATIO in 0.2 0.3 0.4 0.5; do
  echo "--- Byzantine Ratio: ${RATIO} ($(echo "$RATIO * $CLIENTS" | bc)/$CLIENTS clients) ---"

  # FedAVG
  echo "  Running FedAVG..."
  python3 main.py \
    --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --aggr avg --trials 1 --global_epoch $EPOCHS \
    --gpu_id $GPU --bs $BS \
    2>&1 | tee "$RESULTS_DIR/exp2_rop_fedavg_r${RATIO}.log"

  # Krum
  echo "  Running Krum..."
  python3 main.py \
    --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --aggr krum --trials 1 --global_epoch $EPOCHS \
    --gpu_id $GPU --bs $BS \
    2>&1 | tee "$RESULTS_DIR/exp2_rop_krum_r${RATIO}.log"

  # Trimmed-Mean
  echo "  Running Trimmed-Mean..."
  python3 main.py \
    --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --aggr tm --trials 1 --global_epoch $EPOCHS \
    --gpu_id $GPU --bs $BS \
    2>&1 | tee "$RESULTS_DIR/exp2_rop_tm_r${RATIO}.log"

  # FLTG
  echo "  Running FLTG..."
  python3 main.py \
    --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --aggr fltg --trials 1 --global_epoch $EPOCHS \
    --gpu_id $GPU --bs $BS \
    2>&1 | tee "$RESULTS_DIR/exp2_rop_fltg_r${RATIO}.log"

  echo "✓ Completed ratio $RATIO"
  echo ""
done

# ==================================================
# Experiment 3: Different Attack Types (30% Byzantine)
# ==================================================
echo "=== Experiment 3: Different Attack Types (30% Byzantine) ==="
RATIO=0.3

for ATTACK in rop ipm; do
  echo "--- Attack: $ATTACK ---"

  # FedAVG
  echo "  Running FedAVG..."
  python3 main.py \
    --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack $ATTACK \
    --aggr avg --trials 1 --global_epoch $EPOCHS \
    --gpu_id $GPU --bs $BS \
    2>&1 | tee "$RESULTS_DIR/exp3_${ATTACK}_fedavg.log"

  # Krum
  echo "  Running Krum..."
  python3 main.py \
    --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack $ATTACK \
    --aggr krum --trials 1 --global_epoch $EPOCHS \
    --gpu_id $GPU --bs $BS \
    2>&1 | tee "$RESULTS_DIR/exp3_${ATTACK}_krum.log"

  # Trimmed-Mean
  echo "  Running Trimmed-Mean..."
  python3 main.py \
    --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack $ATTACK \
    --aggr tm --trials 1 --global_epoch $EPOCHS \
    --gpu_id $GPU --bs $BS \
    2>&1 | tee "$RESULTS_DIR/exp3_${ATTACK}_tm.log"

  # FLTG
  echo "  Running FLTG..."
  python3 main.py \
    --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack $ATTACK \
    --aggr fltg --trials 1 --global_epoch $EPOCHS \
    --gpu_id $GPU --bs $BS \
    2>&1 | tee "$RESULTS_DIR/exp3_${ATTACK}_fltg.log"

  echo "✓ Completed attack $ATTACK"
  echo ""
done

# ==================================================
# Experiment 4: Non-IID Data Distribution (30% Byzantine, ROP)
# ==================================================
echo "=== Experiment 4: Non-IID Data Distribution (Dirichlet alpha=0.1) ==="
RATIO=0.3
ALPHA=0.1  # Lower alpha = more non-IID

echo "Testing with highly non-IID data (alpha=$ALPHA)..."

# FedAVG
echo "  Running FedAVG..."
python3 main.py \
  --dataset_name $DATASET --nn_name $MODEL \
  --num_client $CLIENTS --traitor $RATIO --attack rop \
  --dataset_dist dirichlet --alpha $ALPHA \
  --aggr avg --trials 1 --global_epoch $EPOCHS \
  --gpu_id $GPU --bs $BS \
  2>&1 | tee "$RESULTS_DIR/exp4_noniid_fedavg.log"

# Krum
echo "  Running Krum..."
python3 main.py \
  --dataset_name $DATASET --nn_name $MODEL \
  --num_client $CLIENTS --traitor $RATIO --attack rop \
  --dataset_dist dirichlet --alpha $ALPHA \
  --aggr krum --trials 1 --global_epoch $EPOCHS \
  --gpu_id $GPU --bs $BS \
  2>&1 | tee "$RESULTS_DIR/exp4_noniid_krum.log"

# Trimmed-Mean
echo "  Running Trimmed-Mean..."
python3 main.py \
  --dataset_name $DATASET --nn_name $MODEL \
  --num_client $CLIENTS --traitor $RATIO --attack rop \
  --dataset_dist dirichlet --alpha $ALPHA \
  --aggr tm --trials 1 --global_epoch $EPOCHS \
  --gpu_id $GPU --bs $BS \
  2>&1 | tee "$RESULTS_DIR/exp4_noniid_tm.log"

# FLTG
echo "  Running FLTG..."
python3 main.py \
  --dataset_name $DATASET --nn_name $MODEL \
  --num_client $CLIENTS --traitor $RATIO --attack rop \
  --dataset_dist dirichlet --alpha $ALPHA \
  --aggr fltg --trials 1 --global_epoch $EPOCHS \
  --gpu_id $GPU --bs $BS \
  2>&1 | tee "$RESULTS_DIR/exp4_noniid_fltg.log"

echo "✓ Completed Non-IID experiments"
echo ""

# ==================================================
# Experiment 5: CIFAR-10 Dataset (Quick test with 30 epochs)
# ==================================================
echo "=== Experiment 5: CIFAR-10 Dataset (30 epochs, 30% Byzantine) ==="
RATIO=0.3
CIFAR_EPOCHS=30
CIFAR_MODEL="resnet20"

echo "Testing with CIFAR-10 (more complex dataset)..."

# Baseline
echo "  Baseline (no attack)..."
python3 main.py \
  --dataset_name cifar10 --nn_name $CIFAR_MODEL \
  --num_client $CLIENTS --traitor 0 \
  --aggr avg --trials 1 --global_epoch $CIFAR_EPOCHS \
  --gpu_id $GPU --bs $BS \
  2>&1 | tee "$RESULTS_DIR/exp5_cifar10_baseline.log"

# FedAVG
echo "  Running FedAVG..."
python3 main.py \
  --dataset_name cifar10 --nn_name $CIFAR_MODEL \
  --num_client $CLIENTS --traitor $RATIO --attack rop \
  --aggr avg --trials 1 --global_epoch $CIFAR_EPOCHS \
  --gpu_id $GPU --bs $BS \
  2>&1 | tee "$RESULTS_DIR/exp5_cifar10_fedavg.log"

# Krum
echo "  Running Krum..."
python3 main.py \
  --dataset_name cifar10 --nn_name $CIFAR_MODEL \
  --num_client $CLIENTS --traitor $RATIO --attack rop \
  --aggr krum --trials 1 --global_epoch $CIFAR_EPOCHS \
  --gpu_id $GPU --bs $BS \
  2>&1 | tee "$RESULTS_DIR/exp5_cifar10_krum.log"

# Trimmed-Mean
echo "  Running Trimmed-Mean..."
python3 main.py \
  --dataset_name cifar10 --nn_name $CIFAR_MODEL \
  --num_client $CLIENTS --traitor $RATIO --attack rop \
  --aggr tm --trials 1 --global_epoch $CIFAR_EPOCHS \
  --gpu_id $GPU --bs $BS \
  2>&1 | tee "$RESULTS_DIR/exp5_cifar10_tm.log"

# FLTG
echo "  Running FLTG..."
python3 main.py \
  --dataset_name cifar10 --nn_name $CIFAR_MODEL \
  --num_client $CLIENTS --traitor $RATIO --attack rop \
  --aggr fltg --trials 1 --global_epoch $CIFAR_EPOCHS \
  --gpu_id $GPU --bs $BS \
  2>&1 | tee "$RESULTS_DIR/exp5_cifar10_fltg.log"

echo "✓ Completed CIFAR-10 experiments"
echo ""

# ==================================================
# Summary
# ==================================================
echo "=================================="
echo "All Experiments Completed!"
echo "=================================="
echo ""
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "Summary of experiments:"
echo "  1. Baseline (50 epochs, no attack)"
echo "  2. Byzantine ratio scaling (20%, 30%, 40%, 50%)"
echo "  3. Different attacks (ROP, IPM)"
echo "  4. Non-IID data (Dirichlet alpha=0.1)"
echo "  5. CIFAR-10 dataset (30 epochs)"
echo ""
echo "Total experiments: ~30 runs"
echo ""
echo "Next step: Run 'python3 analyze_results.py' to generate summary report"