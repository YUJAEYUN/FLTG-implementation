#!/bin/bash

# Focused FLTG Experiments - Addressing the key criticisms
# This runs targeted experiments to test FLTG in scenarios where it should excel

cd FL-Byzantine-Library

echo "=================================="
echo "FLTG Focused Experiments"
echo "=================================="
echo ""

# Create results directory
mkdir -p ../results
RESULTS_DIR="../results"

# Common parameters
DATASET="mnist"
MODEL="mnistnet"
CLIENTS=20
EPOCHS=30  # Balanced: enough for convergence, reasonable time
BS=64
GPU=-1

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Model: $MODEL"
echo "  Clients: $CLIENTS"
echo "  Epochs: $EPOCHS (increased from 10 to 30)"
echo "  Batch Size: $BS"
echo ""

# ==================================================
# Experiment Set 1: Higher Byzantine Ratios
# Paper claims: "robust even with >50% Byzantine clients"
# ==================================================
echo "=== Experiment Set 1: High Byzantine Ratios ==="
echo "Testing FLTG's claim of robustness with >50% malicious clients"
echo ""

for RATIO in 0.3 0.4 0.5; do
  echo "--- Byzantine Ratio: ${RATIO} ($(python3 -c "print(int($RATIO * $CLIENTS))")/$CLIENTS clients) ---"

  echo "  [1/4] FedAVG..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --aggr avg --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/high_byz_${RATIO}_fedavg.log"

  echo "  [2/4] Krum..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --aggr krum --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/high_byz_${RATIO}_krum.log"

  echo "  [3/4] Trimmed-Mean..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --aggr tm --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/high_byz_${RATIO}_tm.log"

  echo "  [4/4] FLTG..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --aggr fltg --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/high_byz_${RATIO}_fltg.log"

  echo "✓ Completed $RATIO"
  echo ""
done

# ==================================================
# Experiment Set 2: Non-IID Data
# Paper claims: "Non-IID aware weighting handles data heterogeneity"
# ==================================================
echo "=== Experiment Set 2: Non-IID Data Distribution ==="
echo "Testing FLTG's Non-IID aware weighting mechanism"
echo ""

RATIO=0.3  # 30% Byzantine

for ALPHA in 0.1 0.5; do
  NONIID_LABEL=$([ "$ALPHA" = "0.1" ] && echo "highly_noniid" || echo "moderate_noniid")
  echo "--- Dirichlet alpha=$ALPHA ($NONIID_LABEL) ---"

  echo "  [1/4] FedAVG..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --dataset_dist dirichlet --alpha $ALPHA \
    --aggr avg --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/noniid_${NONIID_LABEL}_fedavg.log"

  echo "  [2/4] Krum..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --dataset_dist dirichlet --alpha $ALPHA \
    --aggr krum --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/noniid_${NONIID_LABEL}_krum.log"

  echo "  [3/4] Trimmed-Mean..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --dataset_dist dirichlet --alpha $ALPHA \
    --aggr tm --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/noniid_${NONIID_LABEL}_tm.log"

  echo "  [4/4] FLTG..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --dataset_dist dirichlet --alpha $ALPHA \
    --aggr fltg --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/noniid_${NONIID_LABEL}_fltg.log"

  echo "✓ Completed alpha=$ALPHA"
  echo ""
done

# ==================================================
# Experiment Set 3: Different Attack Types
# Paper claims: "robust against various Byzantine attacks"
# ==================================================
echo "=== Experiment Set 3: Multiple Attack Types ==="
echo "Testing robustness against different attack strategies"
echo ""

RATIO=0.3

for ATTACK in rop ipm; do
  echo "--- Attack: $ATTACK ---"

  echo "  [1/4] FedAVG..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack $ATTACK \
    --aggr avg --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/attack_${ATTACK}_fedavg.log"

  echo "  [2/4] Krum..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack $ATTACK \
    --aggr krum --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/attack_${ATTACK}_krum.log"

  echo "  [3/4] Trimmed-Mean..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack $ATTACK \
    --aggr tm --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/attack_${ATTACK}_tm.log"

  echo "  [4/4] FLTG..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack $ATTACK \
    --aggr fltg --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/attack_${ATTACK}_fltg.log"

  echo "✓ Completed $ATTACK"
  echo ""
done

# ==================================================
# Baseline Reference
# ==================================================
echo "=== Baseline Reference (No Attack) ==="
python3 main.py --dataset_name $DATASET --nn_name $MODEL \
  --num_client $CLIENTS --traitor 0 \
  --aggr avg --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
  2>&1 | grep "Epoch" | tee "$RESULTS_DIR/baseline_30ep.log"

echo "✓ Baseline completed"
echo ""

echo "=================================="
echo "Experiments Completed!"
echo "=================================="
echo ""
echo "Total runs: 26 experiments"
echo "  - 3 Byzantine ratios × 4 methods = 12 runs"
echo "  - 2 Non-IID levels × 4 methods = 8 runs"
echo "  - 2 Attack types × 4 methods = 8 runs (overlap with ratio 30%)"
echo "  - 1 Baseline = 1 run"
echo ""
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "Next: Run './analyze_focused_results.py' to see summary"