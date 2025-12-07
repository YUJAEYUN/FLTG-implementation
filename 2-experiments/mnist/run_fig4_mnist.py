import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_list_floats(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def run_with_log(cmd, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            if "Epoch" in line:
                print(line, end="")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(cmd)}")


def build_base(args: argparse.Namespace, alpha: float) -> List[str]:
    return [
        sys.executable,
        "main.py",
        "--dataset_name",
        args.dataset,
        "--dataset_dist",
        "dirichlet",
        "--alpha",
        str(alpha),
        "--nn_name",
        args.model,
        "--num_client",
        str(args.clients),
        "--global_epoch",
        str(args.epochs),
        "--gpu_id",
        str(args.gpu_id),
        "--bs",
        str(args.batch_size),
    ]


def run_baseline(args, lib_dir: Path, out_dir: Path, alpha: float) -> None:
    log = out_dir / f"alpha_{alpha}/baseline.log"
    cmd = build_base(args, alpha) + ["--traitor", "0", "--aggr", "avg"]
    run_with_log(cmd, lib_dir, log)


def run_attack_grid(args, lib_dir: Path, out_dir: Path, alpha: float) -> None:
    base_cmd = build_base(args, alpha)
    for ratio in args.ratios:
        for aggr in args.aggregators:
            log_file = out_dir / f"alpha_{alpha}" / f"ratio_{ratio}" / f"{aggr}.log"
            cmd = list(base_cmd)
            cmd.extend(
                [
                    "--traitor",
                    str(ratio),
                    "--attack",
                    args.attack,
                    "--aggr",
                    aggr,
                ]
            )
            run_with_log(cmd, lib_dir, log_file)


def main():
    parser = argparse.ArgumentParser(description="Run MNIST experiments approximating Fig.4.")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--model", default="mnistnet")
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--alpha_levels", type=parse_list_floats, default="0.1,0.5,1.0",
                        help="Dirichlet alpha values to mimic non-IID degrees")
    parser.add_argument("--bias", type=float, default=0.1, help="Root dataset bias probability (informational)")
    parser.add_argument("--ratios", type=parse_list_floats, default="0.2,0.5,0.8,0.9,0.95")
    parser.add_argument(
        "--aggregators",
        type=lambda x: [s.strip() for s in x.split(",") if s.strip()],
        default="fl_trust,fltg",
    )
    parser.add_argument("--attack", default="rop", help="Attack used to approximate adaptive poisoning")
    parser.add_argument("--results_root", default="results_fig4_mnist")
    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = Path(__file__).resolve().parent
    lib_dir = root_dir / "FL-Byzantine-Library"
    if not lib_dir.exists():
        raise FileNotFoundError("FL-Byzantine-Library not found next to this script.")

    out_dir = Path(args.results_root) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("====================================")
    print("MNIST Fig.4 Approximate Experiments")
    print(f"Dataset={args.dataset}, Model={args.model}")
    print(f"Alpha levels={args.alpha_levels} (non-IID degrees)")
    print(f"Bias probability (fixed)={args.bias}")
    print(f"Malicious ratios={args.ratios}")
    print(f"Aggregators={args.aggregators}")
    print(f"Attack={args.attack}")
    print(f"Logs -> {out_dir}")
    print("====================================\n")

    for alpha in args.alpha_levels:
        print(f"=== Dirichlet alpha {alpha} ===")
        run_baseline(args, lib_dir, out_dir, alpha)
        run_attack_grid(args, lib_dir, out_dir, alpha)
        print(f"Completed alpha {alpha}\n")

    print(f"All experiments completed. Logs saved to {out_dir}")


if __name__ == "__main__":
    main()
