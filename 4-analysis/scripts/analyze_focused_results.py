#!/usr/bin/env python3
"""
Analyze focused experimental results
"""

import os
import re

def parse_log(filepath):
    """Extract final epoch accuracy from log"""
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find last epoch
    for line in reversed(lines):
        match = re.search(r'Epoch\s+\d+\s+Accuracy\s+([\d.]+)', line)
        if match:
            return float(match.group(1))
    return None

def print_table(title, data, baseline=None):
    """Print formatted comparison table"""
    print(f"\n{title}")
    print("=" * 70)
    print(f"{'Method':<15} {'Accuracy':<12} {'vs Baseline':<15} {'Rank'}")
    print("-" * 70)

    # Sort by accuracy (descending)
    sorted_data = sorted(data.items(), key=lambda x: x[1] if x[1] else 0, reverse=True)

    for rank, (method, acc) in enumerate(sorted_data, 1):
        if acc is None:
            print(f"{method:<15} {'N/A':<12} {'N/A':<15} {'-'}")
        else:
            diff = f"{acc - baseline:+.2f}%p" if baseline else "N/A"
            marker = " ⭐" if rank == 1 else ""
            print(f"{method:<15} {acc:.2f}%{'':<6} {diff:<15} {rank}{marker}")

def main():
    results_dir = "results"

    if not os.path.exists(results_dir):
        print("Error: results directory not found!")
        return

    print("\n" + "=" * 70)
    print(" FLTG COMPREHENSIVE EXPERIMENTAL RESULTS")
    print("=" * 70)

    # Baseline
    baseline = parse_log(f"{results_dir}/baseline_30ep.log")
    if baseline:
        print(f"\n📊 Baseline (No Attack, 30 epochs): {baseline:.2f}%")

    # ===== Experiment 1: High Byzantine Ratios =====
    print("\n\n🔴 HIGH BYZANTINE RATIO EXPERIMENTS")
    print("Testing paper's claim: 'robust with >50% Byzantine clients'\n")

    for ratio in [0.3, 0.4, 0.5]:
        data = {
            'FedAVG': parse_log(f"{results_dir}/high_byz_{ratio}_fedavg.log"),
            'Krum': parse_log(f"{results_dir}/high_byz_{ratio}_krum.log"),
            'Trimmed-Mean': parse_log(f"{results_dir}/high_byz_{ratio}_tm.log"),
            'FLTG': parse_log(f"{results_dir}/high_byz_{ratio}_fltg.log"),
        }

        byz_count = int(ratio * 20)
        print_table(
            f"Byzantine Ratio: {ratio:.0%} ({byz_count}/20 clients) - ROP Attack",
            data,
            baseline
        )

    # ===== Experiment 2: Non-IID Data =====
    print("\n\n🟡 NON-IID DATA EXPERIMENTS")
    print("Testing paper's claim: 'Non-IID aware weighting handles heterogeneity'\n")

    for alpha, label in [('0.1', 'highly_noniid'), ('0.5', 'moderate_noniid')]:
        data = {
            'FedAVG': parse_log(f"{results_dir}/noniid_{label}_fedavg.log"),
            'Krum': parse_log(f"{results_dir}/noniid_{label}_krum.log"),
            'Trimmed-Mean': parse_log(f"{results_dir}/noniid_{label}_tm.log"),
            'FLTG': parse_log(f"{results_dir}/noniid_{label}_fltg.log"),
        }

        noniid_desc = "Highly Non-IID" if alpha == '0.1' else "Moderate Non-IID"
        print_table(
            f"{noniid_desc} (Dirichlet α={alpha}, 30% Byzantine, ROP)",
            data,
            baseline
        )

    # ===== Experiment 3: Different Attacks =====
    print("\n\n🟢 MULTIPLE ATTACK TYPE EXPERIMENTS")
    print("Testing paper's claim: 'robust against various attacks'\n")

    for attack in ['rop', 'ipm']:
        data = {
            'FedAVG': parse_log(f"{results_dir}/attack_{attack}_fedavg.log"),
            'Krum': parse_log(f"{results_dir}/attack_{attack}_krum.log"),
            'Trimmed-Mean': parse_log(f"{results_dir}/attack_{attack}_tm.log"),
            'FLTG': parse_log(f"{results_dir}/attack_{attack}_fltg.log"),
        }

        attack_name = "ROP (Relocated Orthogonal Perturbation)" if attack == 'rop' else "IPM (Inner Product Manipulation)"
        print_table(
            f"{attack_name} (30% Byzantine)",
            data,
            baseline
        )

    # ===== Summary =====
    print("\n\n" + "=" * 70)
    print(" SUMMARY OF KEY FINDINGS")
    print("=" * 70)

    # Count wins
    wins = {'FedAVG': 0, 'Krum': 0, 'Trimmed-Mean': 0, 'FLTG': 0}

    # Check all experiments
    experiments = []

    # High Byzantine
    for ratio in [0.3, 0.4, 0.5]:
        results = {
            'FedAVG': parse_log(f"{results_dir}/high_byz_{ratio}_fedavg.log"),
            'Krum': parse_log(f"{results_dir}/high_byz_{ratio}_krum.log"),
            'Trimmed-Mean': parse_log(f"{results_dir}/high_byz_{ratio}_tm.log"),
            'FLTG': parse_log(f"{results_dir}/high_byz_{ratio}_fltg.log"),
        }
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            winner = max(valid_results, key=valid_results.get)
            wins[winner] += 1
            experiments.append((f"{ratio:.0%} Byzantine", winner, valid_results[winner]))

    # Non-IID
    for alpha, label in [('0.1', 'highly_noniid'), ('0.5', 'moderate_noniid')]:
        results = {
            'FedAVG': parse_log(f"{results_dir}/noniid_{label}_fedavg.log"),
            'Krum': parse_log(f"{results_dir}/noniid_{label}_krum.log"),
            'Trimmed-Mean': parse_log(f"{results_dir}/noniid_{label}_tm.log"),
            'FLTG': parse_log(f"{results_dir}/noniid_{label}_fltg.log"),
        }
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            winner = max(valid_results, key=valid_results.get)
            wins[winner] += 1
            noniid_desc = "Highly Non-IID" if alpha == '0.1' else "Moderate Non-IID"
            experiments.append((noniid_desc, winner, valid_results[winner]))

    # Attacks
    for attack in ['rop', 'ipm']:
        results = {
            'FedAVG': parse_log(f"{results_dir}/attack_{attack}_fedavg.log"),
            'Krum': parse_log(f"{results_dir}/attack_{attack}_krum.log"),
            'Trimmed-Mean': parse_log(f"{results_dir}/attack_{attack}_tm.log"),
            'FLTG': parse_log(f"{results_dir}/attack_{attack}_fltg.log"),
        }
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            winner = max(valid_results, key=valid_results.get)
            wins[winner] += 1
            experiments.append((f"{attack.upper()} attack", winner, valid_results[winner]))

    print("\n📈 Win Count (Best performer in each scenario):")
    sorted_wins = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    for method, count in sorted_wins:
        bar = "█" * count
        print(f"  {method:<15} {count:>2} wins  {bar}")

    print("\n🏆 Individual Scenario Winners:")
    for scenario, winner, acc in experiments:
        print(f"  {scenario:<25} → {winner:<15} ({acc:.2f}%)")

    print("\n💡 Key Insights:")

    # Check if FLTG wins in high Byzantine scenarios
    high_byz_fltg_wins = sum(1 for s, w, _ in experiments if "Byzantine" in s and "50%" in s and w == "FLTG")
    if high_byz_fltg_wins > 0:
        print("  ✓ FLTG shows strength in high Byzantine ratio scenarios (>50%)")
    else:
        print("  ✗ FLTG does NOT dominate in high Byzantine scenarios as paper claims")

    # Check if FLTG wins in Non-IID
    noniid_fltg_wins = sum(1 for s, w, _ in experiments if "Non-IID" in s and w == "FLTG")
    if noniid_fltg_wins >= 1:
        print("  ✓ FLTG's Non-IID aware weighting provides advantage")
    else:
        print("  ✗ FLTG's Non-IID aware weighting does NOT show clear benefit")

    # Overall winner
    overall_winner = sorted_wins[0][0]
    if overall_winner == "FLTG":
        print(f"  ✓ FLTG is the overall winner across scenarios")
    else:
        print(f"  ✗ {overall_winner} outperforms FLTG overall")

    print("\n" + "=" * 70)
    print(" CONCLUSION")
    print("=" * 70)

    if wins['FLTG'] >= len(experiments) // 2:
        print("\n✅ FLTG validates paper claims in this experimental setting")
        print("   The algorithm shows robust performance across diverse scenarios.")
    else:
        print("\n⚠️  FLTG does NOT fully validate paper claims")
        print(f"   FLTG won {wins['FLTG']}/{len(experiments)} scenarios.")
        print(f"   {sorted_wins[0][0]} shows stronger overall performance.")
        print("\n   Possible reasons:")
        print("   1. Implementation differences from paper")
        print("   2. Hyperparameter tuning needed")
        print("   3. Dataset/model complexity insufficient to show advantage")
        print("   4. Longer training (>30 epochs) may be needed")

    print("\n")

if __name__ == "__main__":
    main()