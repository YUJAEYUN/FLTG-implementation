import os
import subprocess
import csv
import re

# ==== 공통 설정 ====
PYTHON = "python"

# === 데이터셋 / 모델 변경 ===
DATASET = "cifar10"       # mnist -> cifar10
NN_NAME = "simplecifar"      # mnistnet -> resnet20

NUM_CLIENT = 100
ATTACK = "minmax"

# 논문 Fig.3와 맞추는 분포 설정
DIST_OPT = ["--dataset_dist", "root_bias"]
NONIID_OPT = ["--non_iid", "0.5"]   # q = 0.5

# CIFAR-10은 MNIST보다 어려우니까 epoch을 조금 늘려도 좋음
GLOBAL_EPOCH = 5       # 20으로 둬도 되지만 50 정도가 논문 느낌에 더 가까움
TRIALS = 1
BATCH_SIZE = 64
GPU_ID = -1

BIAS_LIST = [0.1, 0.5, 0.8]
TRAITOR_LIST = [0.2, 0.5, 0.8, 0.95]
AGGR_LIST = ["fltg", "fl_trust"]

# 저장 경로(로그/CSV/npy 각각 분리)
LOG_DIR = "logs_fig3_cifar10_client100_simplecifar_epoch5"
RESULT_CSV = "results_fig3_cifar10_client100_simplecifar_epoch5.csv"
RESULT_DIR_NPY = "Results_fig3_cifar10_client100_simplecifar_epoch5"


# ===============================
# 정확도 추출 함수
# ===============================
def extract_acc(log_text: str):
    lines = log_text.splitlines()
    acc_candidates = []

    for line in lines:
        m = re.search(r"Epoch\s+\d+\s+Accuracy\s+([0-9.]+)", line)
        if m:
            acc_candidates.append(m.group(1))

    if acc_candidates:
        return acc_candidates[-1]

    return None


# ===============================
# 한 실험 실행
# ===============================
def run_experiment(bias, traitor, aggr, attack, extra_tag=""):
    if extra_tag:
        run_name = f"bias{bias}_traitor{traitor}_{aggr}_{extra_tag}"
    else:
        run_name = f"bias{bias}_traitor{traitor}_{aggr}"

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, run_name + ".log")

    print(f"\n>>> Running: bias={bias}, traitor={traitor}, aggr={aggr}, attack={attack}")
    print(f"    로그 파일: {log_path}")

    cmd = [
        PYTHON, "main.py",
        "--dataset_name", DATASET,
        "--nn_name", NN_NAME,
        "--num_client", str(NUM_CLIENT),
        "--traitor", str(traitor),
        "--attack", attack,
        *DIST_OPT,
        *NONIID_OPT,
        "--bias_prob", str(bias),
        "--aggr", aggr,
        "--trials", str(TRIALS),
        "--global_epoch", str(GLOBAL_EPOCH),
        "--gpu_id", str(GPU_ID),
        "--bs", str(BATCH_SIZE),

        # ★ main.py에게 npy 저장 폴더 전달
        "--result_dir", RESULT_DIR_NPY
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)

    acc = extract_acc(result.stdout)
    print(f"    원본 Accuracy = {acc}")

    acc_float = float(acc) if acc is not None else None
    if acc_float is not None:
        acc_float = acc_float / 100.0

    return round(acc_float, 3) if acc_float is not None else None


# ===============================
# 메인 실행
# ===============================
def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # CSV 헤더
    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bias_prob", "traitor", "aggr", "test_acc"])

    print("=== Fig.3 실험 시작 ===")

    # 1. 공격 있는 실험
    for bias in BIAS_LIST:
        for traitor in TRAITOR_LIST:
            for aggr in AGGR_LIST:
                acc = run_experiment(bias, traitor, aggr, ATTACK)
                with open(RESULT_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([bias, traitor, aggr, acc])

    # 2. FedAvg no-attack
    for bias in BIAS_LIST:
        acc = run_experiment(bias, 0.0, "avg", "none", extra_tag="no_attack")
        with open(RESULT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([bias, 0.0, "avg_no_attack", acc])

    print("\n=== 모든 Fig.3 실험 종료 ===")


if __name__ == "__main__":
    main()