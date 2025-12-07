#!/bin/bash

# EXTREME Non-IID Experiments - Making MNIST Actually Challenging
# Goal: Create scenarios where defenses MUST show clear differences

cd FL-Byzantine-Library

echo "=========================================="
echo "FLTG EXTREME Non-IID Experiments"
echo "=========================================="
echo ""
echo "Strategy: Make MNIST hard by extreme data heterogeneity"
echo ""

mkdir -p ../results
RESULTS_DIR="../results"

DATASET="mnist"
MODEL="mnistnet"
CLIENTS=20
EPOCHS=20  # Reduced - differences appear early in extreme scenarios
BS=64
GPU=-1

echo "Configuration:"
echo "  Dataset: MNIST"
echo "  Model: MNISTNET"
echo "  Clients: 20"
echo "  Epochs: 20 (reduced for faster iteration)"
echo "  Batch Size: 64"
echo ""

# ==================================================
# Experiment 1: Extreme Non-IID with High Byzantine
# Combining worst-case scenarios
# ==================================================
echo "=== Experiment 1: EXTREME Non-IID + High Byzantine ==="
echo "Each client gets only 1-2 digit classes + 50% are malicious"
echo ""

# Baseline for comparison
echo "--- Baseline: IID, No Attack ---"
python3 main.py --dataset_name $DATASET --nn_name $MODEL \
  --num_client $CLIENTS --traitor 0 --dataset_dist iid \
  --aggr avg --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
  2>&1 | grep "Epoch" | tee "$RESULTS_DIR/extreme_baseline.log"
echo "✓ Baseline completed"
echo ""

# Extreme scenarios
for ALPHA in 0.01 0.1; do
  LABEL=$([ "$ALPHA" = "0.01" ] && echo "ultra_extreme" || echo "extreme")

  echo "--- Non-IID Level: $LABEL (alpha=$ALPHA) ---"

  for RATIO in 0.3 0.5; do
    echo "  Byzantine Ratio: ${RATIO} ($(python3 -c "print(int($RATIO * $CLIENTS))")/$CLIENTS clients)"

    echo "    [1/4] FedAVG..."
    python3 main.py --dataset_name $DATASET --nn_name $MODEL \
      --num_client $CLIENTS --traitor $RATIO --attack rop \
      --dataset_dist dirichlet --alpha $ALPHA \
      --aggr avg --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
      2>&1 | grep "Epoch" | tee "$RESULTS_DIR/${LABEL}_byz${RATIO}_fedavg.log"

    echo "    [2/4] Krum..."
    python3 main.py --dataset_name $DATASET --nn_name $MODEL \
      --num_client $CLIENTS --traitor $RATIO --attack rop \
      --dataset_dist dirichlet --alpha $ALPHA \
      --aggr krum --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
      2>&1 | grep "Epoch" | tee "$RESULTS_DIR/${LABEL}_byz${RATIO}_krum.log"

    echo "    [3/4] Trimmed-Mean..."
    python3 main.py --dataset_name $DATASET --nn_name $MODEL \
      --num_client $CLIENTS --traitor $RATIO --attack rop \
      --dataset_dist dirichlet --alpha $ALPHA \
      --aggr tm --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
      2>&1 | grep "Epoch" | tee "$RESULTS_DIR/${LABEL}_byz${RATIO}_tm.log"

    echo "    [4/4] FLTG..."
    python3 main.py --dataset_name $DATASET --nn_name $MODEL \
      --num_client $CLIENTS --traitor $RATIO --attack rop \
      --dataset_dist dirichlet --alpha $ALPHA \
      --aggr fltg --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
      2>&1 | grep "Epoch" | tee "$RESULTS_DIR/${LABEL}_byz${RATIO}_fltg.log"

    echo "  ✓ Completed ${RATIO}"
    echo ""
  done
done

# ==================================================
# Experiment 2: Sort-and-Partition (Class Imbalance)
# Each client gets only specific classes
# ==================================================
echo "=== Experiment 2: Class Imbalance (2 classes per client) ==="
echo "Simulating real-world scenarios like hospitals with specialized departments"
echo ""

for RATIO in 0.3 0.5; do
  echo "--- Byzantine Ratio: ${RATIO} ---"

  echo "  [1/4] FedAVG..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --dataset_dist sort_part --numb_cls_usr 2 \
    --aggr avg --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/class_imbal_byz${RATIO}_fedavg.log"

  echo "  [2/4] Krum..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --dataset_dist sort_part --numb_cls_usr 2 \
    --aggr krum --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/class_imbal_byz${RATIO}_krum.log"

  echo "  [3/4] Trimmed-Mean..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --dataset_dist sort_part --numb_cls_usr 2 \
    --aggr tm --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/class_imbal_byz${RATIO}_tm.log"

  echo "  [4/4] FLTG..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack rop \
    --dataset_dist sort_part --numb_cls_usr 2 \
    --aggr fltg --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/class_imbal_byz${RATIO}_fltg.log"

  echo "✓ Completed ${RATIO}"
  echo ""
done

# ==================================================
# Experiment 3: Multiple Attack Types in Extreme Non-IID
# ==================================================
echo "=== Experiment 3: Different Attacks in Extreme Non-IID ==="
echo ""

ALPHA=0.01  # Ultra extreme Non-IID
RATIO=0.5   # 50% Byzantine

for ATTACK in rop ipm; do
  echo "--- Attack: $ATTACK (50% Byzantine, α=0.01) ---"

  echo "  [1/4] FedAVG..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack $ATTACK \
    --dataset_dist dirichlet --alpha $ALPHA \
    --aggr avg --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/extreme_${ATTACK}_fedavg.log"

  echo "  [2/4] Krum..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack $ATTACK \
    --dataset_dist dirichlet --alpha $ALPHA \
    --aggr krum --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/extreme_${ATTACK}_krum.log"

  echo "  [3/4] Trimmed-Mean..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack $ATTACK \
    --dataset_dist dirichlet --alpha $ALPHA \
    --aggr tm --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/extreme_${ATTACK}_tm.log"

  echo "  [4/4] FLTG..."
  python3 main.py --dataset_name $DATASET --nn_name $MODEL \
    --num_client $CLIENTS --traitor $RATIO --attack $ATTACK \
    --dataset_dist dirichlet --alpha $ALPHA \
    --aggr fltg --trials 1 --global_epoch $EPOCHS --gpu_id $GPU --bs $BS \
    2>&1 | grep "Epoch" | tee "$RESULTS_DIR/extreme_${ATTACK}_fltg.log"

  echo "✓ Completed $ATTACK"
  echo ""
done

echo "=========================================="
echo "Extreme Experiments Completed!"
echo "=========================================="
echo ""
echo "Total: 21 experiments"
echo "  - 2 Non-IID levels × 2 Byzantine ratios × 4 methods = 16"
echo "  - 1 Class imbalance × 2 Byzantine ratios × 4 methods = 8 (overlap)"
echo "  - 2 Attack types × 4 methods = 8 (overlap)"
echo "  - 1 Baseline = 1"
echo ""
echo "Estimated time: ~1.5-2 hours"
echo ""
echo "Next: python3 analyze_extreme_results.py"