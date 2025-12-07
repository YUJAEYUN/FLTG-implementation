import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

EPOCH_PATTERN = re.compile(r"Epoch\s+\d+\s+Accuracy\s+([0-9]+(?:\.[0-9]+)?)")


def parse_accuracy(log_path: Path) -> float:
    last_acc = None
    with log_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = EPOCH_PATTERN.search(line)
            if match:
                last_acc = float(match.group(1))
    if last_acc is None:
        raise ValueError(f"No accuracy found in {log_path}")
    return last_acc


def collect_results(run_dir: Path) -> Dict[float, Dict[float, Dict[str, float]]]:
    data: Dict[float, Dict[float, Dict[str, float]]] = {}
    for alpha_dir in sorted(run_dir.glob("alpha_*")):
        alpha = float(alpha_dir.name.split("_")[-1])
        data[alpha] = {}
        for ratio_dir in sorted((d for d in alpha_dir.glob("ratio_*") if d.is_dir())):
            ratio = float(ratio_dir.name.split("_")[-1])
            data[alpha][ratio] = {}
            for log_path in ratio_dir.glob("*.log"):
                aggregator = log_path.stem
                data[alpha][ratio][aggregator] = parse_accuracy(log_path)
        baseline = alpha_dir / "baseline.log"
        if baseline.exists():
            data[alpha]["baseline"] = {"avg": parse_accuracy(baseline)}
    return data


def plot_results(data: Dict[float, Dict[float, Dict[str, float]]], aggregators: List[str], output: Path) -> None:
    alphas = sorted(k for k in data.keys())
    fig, axes = plt.subplots(1, len(alphas), figsize=(6 * len(alphas), 4), sharey=True)
    if len(alphas) == 1:
        axes = [axes]
    for ax, alpha in zip(axes, alphas):
        ratios = sorted(k for k in data[alpha].keys() if k != "baseline")
        for aggr in aggregators:
            accs = [data[alpha][ratio].get(aggr, float("nan")) for ratio in ratios]
            ax.plot(ratios, accs, marker="o", label=aggr)
        baseline = data[alpha].get("baseline", {}).get("avg")
        if baseline is not None:
            ax.axhline(baseline, linestyle="--", color="green", label="FedAvg no attack")
        ax.set_title(f"Dirichlet alpha = {alpha}")
        ax.set_xlabel("Fraction of malicious clients")
        ax.set_ylabel("Testing accuracy")
        ax.set_ylim(0, 100)
        ax.grid(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot MNIST Fig.4 style results.")
    parser.add_argument("--run_dir", required=True, help="results_fig4_mnist/<timestamp> directory")
    parser.add_argument("--aggregators", default="fl_trust,fltg")
    parser.add_argument("--output", default="fig4_mnist.png")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"{run_dir} not found.")
    aggregators = [a.strip() for a in args.aggregators.split(",") if a.strip()]

    data = collect_results(run_dir)
    plot_results(data, aggregators, Path(args.output))
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
