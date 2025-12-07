import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def _load_data_loader():
    """Import data_loader from the original library without modifying it."""
    project_root = Path(__file__).resolve().parent / "FL-Byzantine-Library"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import data_loader as dl  # type: ignore

    return dl


def generate_dirichlet_split(dataset_name: str, num_clients: int, alpha: float, seed: int) -> Dict[int, List[int]]:
    dl = _load_data_loader()
    args = argparse.Namespace(
        dataset_name=dataset_name,
        num_client=num_clients,
        alpha=alpha,
        dataset_dist="dirichlet",
    )
    trainset, _ = dl.get_dataset(args)
    rng_state = np.random.get_state()
    np.random.seed(seed)
    indices, _ = dl.dirichlet_dist(trainset.targets, args)
    np.random.set_state(rng_state)
    return {int(k): list(map(int, v)) for k, v in indices.items()}


def sample_biased_root_indices(dataset_name: str, bias: float, root_size: int, seed: int) -> List[int]:
    dl = _load_data_loader()
    args = argparse.Namespace(dataset_name=dataset_name)
    trainset, _ = dl.get_dataset(args)
    targets = np.asarray(trainset.targets, dtype=int)
    classes = np.unique(targets)
    rng = random.Random(seed)

    class_probs = {cls: (1 - bias) / (len(classes) - 1) for cls in classes}
    class_probs[classes[0]] = bias
    per_class_indices = {cls: np.where(targets == cls)[0].tolist() for cls in classes}
    for indices in per_class_indices.values():
        rng.shuffle(indices)

    chosen = []
    while len(chosen) < root_size:
        for cls in classes:
            if per_class_indices[cls] and rng.random() < class_probs[cls]:
                chosen.append(per_class_indices[cls].pop())
                if len(chosen) == root_size:
                    break
    return chosen


def main():
    parser = argparse.ArgumentParser(description="Generate CIFAR dirichlet splits and biased root datasets.")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--bias", type=float, default=0.1)
    parser.add_argument("--root_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="new_cifar_bias_configs")
    args = parser.parse_args()

    dirichlet = generate_dirichlet_split(args.dataset, args.clients, args.alpha, args.seed)
    root_indices = sample_biased_root_indices(args.dataset, args.bias, args.root_size, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"dirichlet_{args.alpha}.json").write_text(json.dumps(dirichlet, indent=2), encoding="utf-8")
    (output_dir / f"root_bias_{args.bias}.json").write_text(json.dumps(root_indices), encoding="utf-8")
    print(f"Saved dirichlet split and root bias indices to {output_dir}")


if __name__ == "__main__":
    main()
