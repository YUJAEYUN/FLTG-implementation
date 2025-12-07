import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import List



def parse_list(value: str) -> List[float]:
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


def build_base(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        "main.py",
        "--dataset_name",
        args.dataset,
        "--dataset_dist",
        "dirichlet",
        "--alpha",
        str(args.alpha),
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


def run_baseline(args, lib_dir: Path, out_dir: Path, bias: float) -> None:
    log = out_dir / f"bias_{bias}/baseline.log"
    bias_file = (Path(args.root_bias_dir) / f"root_bias_{bias}.json").resolve()
    cmd = build_base(args) + ["--traitor", "0", "--aggr", "avg", "--root_bias_path", str(bias_file)]
    run_with_log(cmd, lib_dir, log)



def run_attack_grid(args, lib_dir: Path, out_dir: Path, bias: float) -> None:
    bias_file = (Path(args.root_bias_dir) / f"root_bias_{bias}.json").resolve()
    base_cmd = build_base(args) + ["--root_bias_path", str(bias_file)]
    for ratio in args.ratios:
        for aggr in args.aggregators:
            tag = out_dir / f"bias_{bias}" / f"ratio_{ratio}" / f"{aggr}.log"
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
            run_with_log(cmd, lib_dir, tag)


def main():
    parser = argparse.ArgumentParser(description="Run MNIST experiments approximating Fig.3.")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", default="resnet20")
    parser.add_argument("--clients", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for client skew")
    parser.add_argument("--bias_levels", type=parse_list, default="0.1,0.5,0.8")
    parser.add_argument("--ratios", type=parse_list, default="0.2,0.5,0.8,0.95")
    parser.add_argument("--aggregators", type=lambda x: [s.strip() for s in x.split(",") if s.strip()],
                        default="fl_trust,fltg")
    parser.add_argument("--attack", default="minmax", help="Attack used to approximate adaptive poisoning")
    parser.add_argument("--results_root", default="results_fig3_mnist")
    parser.add_argument("--root_bias_dir", default="mnist_bias_configs",
                        help="Directory containing root_bias_<bias>.json files")

    args = parser.parse_args()

    args.ratios = [round(r, 2) for r in args.ratios]
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    root_dir = Path(__file__).resolve().parent
    lib_dir = root_dir / "FL-Byzantine-Library"
    if not lib_dir.exists():
        raise FileNotFoundError("FL-Byzantine-Library not found next to this script.")

    out_dir = Path(args.results_root) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("====================================")
    print("MNIST Fig.3 Approximate Experiments")
    print(f"Dataset={args.dataset}, Model={args.model}")
    print(f"Clients={args.clients}, Epochs={args.epochs}, Batch size={args.batch_size}")
    print(f"Dirichlet alpha={args.alpha}")
    print(f"Bias levels={args.bias_levels}")
    print(f"Ratios={args.ratios}")
    print(f"Aggregators={args.aggregators}")
    print(f"Attack={args.attack}")
    print(f"Logs -> {out_dir}")
    print("====================================\n")

    for bias in args.bias_levels:
        print(f"=== Bias probability {bias} ===")
        run_baseline(args, lib_dir, out_dir, bias)
        run_attack_grid(args, lib_dir, out_dir, bias)
        print(f"Completed bias {bias}\n")

    print(f"All experiments completed. Logs saved to {out_dir}")


if __name__ == "__main__":
    main()
