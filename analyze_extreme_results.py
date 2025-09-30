#!/usr/bin/env python3
"""
Analyze EXTREME Non-IID experimental results
Focus: Finding scenarios where FLTG should show its strength
"""

import os
import re

def parse_log(filepath):
    """Extract final accuracy"""
    if not os.path.exists(filepath):
        return None, None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find first and last epoch
    first_acc, last_acc = None, None

    for line in lines:
        match = re.search(r'Epoch\s+\d+\s+Accuracy\s+([\d.]+)', line)
        if match:
            acc = float(match.group(1))
            if first_acc is None:
                first_acc = acc
            last_acc = acc

    return first_acc, last_acc

def print_comparison(title, data, baseline=None):
    """Print formatted comparison with convergence info"""
    print(f"\n{title}")
    print("=" * 90)
    print(f"{'Method':<15} {'Epoch 1':<12} {'Final (E20)':<12} {'Improvement':<12} {'vs Baseline':<15} {'Rank'}")
    print("-" * 90)

    # Calculate improvements and sort by final accuracy
    results = []
    for method, (first, last) in data.items():
        if first is not None and last is not None:
            improvement = last - first
            results.append((method, first, last, improvement))

    results.sort(key=lambda x: x[2], reverse=True)

    for rank, (method, first, last, improvement) in enumerate(results, 1):
        vs_baseline = f"{last - baseline:+.2f}%p" if baseline else "N/A"
        marker = " ⭐" if rank == 1 else ""
        color = "🟢" if improvement > 10 else "🟡" if improvement > 5 else "🔴"
        print(f"{method:<15} {first:>5.2f}%{'':<5} {last:>5.2f}%{'':<5} {color} {improvement:+.2f}%p{'':<3} {vs_baseline:<15} {rank}{marker}")

    # Add no-data entries
    for method, (first, last) in data.items():
        if first is None:
            print(f"{method:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'-'}")

def main():
    results_dir = "results"

    if not os.path.exists(results_dir):
        print("Error: results directory not found!")
        return

    print("\n" + "=" * 90)
    print(" 🔥 EXTREME Non-IID EXPERIMENTAL RESULTS 🔥")
    print("=" * 90)
    print("\nGoal: Find scenarios where FLTG's Non-IID aware weighting shows advantage\n")

    # Baseline
    _, baseline = parse_log(f"{results_dir}/extreme_baseline.log")
    if baseline:
        print(f"📊 Baseline (IID, No Attack, 20 epochs): {baseline:.2f}%")

    # ===== Experiment 1: Extreme Non-IID + High Byzantine =====
    print("\n\n🔥 EXTREME Non-IID EXPERIMENTS")
    print("Paper's claim: 'Non-IID aware weighting handles data heterogeneity'\n")

    # Ultra Extreme (alpha=0.01)
    print("### Ultra Extreme Non-IID (Dirichlet α=0.01) ###")
    print("Each client sees mostly 1-2 digit classes only!\n")

    for ratio in [0.3, 0.5]:
        data = {
            'FedAVG': parse_log(f"{results_dir}/ultra_extreme_byz{ratio}_fedavg.log"),
            'Krum': parse_log(f"{results_dir}/ultra_extreme_byz{ratio}_krum.log"),
            'Trimmed-Mean': parse_log(f"{results_dir}/ultra_extreme_byz{ratio}_tm.log"),
            'FLTG': parse_log(f"{results_dir}/ultra_extreme_byz{ratio}_fltg.log"),
        }

        byz_count = int(ratio * 20)
        print_comparison(
            f"🔴 Ultra Extreme Non-IID + {byz_count}/20 Byzantine ({ratio:.0%}) - ROP Attack",
            data,
            baseline
        )

    # Extreme (alpha=0.1)
    print("\n\n### Extreme Non-IID (Dirichlet α=0.1) ###")

    for ratio in [0.3, 0.5]:
        data = {
            'FedAVG': parse_log(f"{results_dir}/extreme_byz{ratio}_fedavg.log"),
            'Krum': parse_log(f"{results_dir}/extreme_byz{ratio}_krum.log"),
            'Trimmed-Mean': parse_log(f"{results_dir}/extreme_byz{ratio}_tm.log"),
            'FLTG': parse_log(f"{results_dir}/extreme_byz{ratio}_fltg.log"),
        }

        byz_count = int(ratio * 20)
        print_comparison(
            f"🟠 Extreme Non-IID + {byz_count}/20 Byzantine ({ratio:.0%}) - ROP Attack",
            data,
            baseline
        )

    # ===== Experiment 2: Class Imbalance =====
    print("\n\n🟡 CLASS IMBALANCE EXPERIMENTS")
    print("Each client gets only 2 classes (e.g., only digits 0&1, or 2&3, etc.)\n")

    for ratio in [0.3, 0.5]:
        data = {
            'FedAVG': parse_log(f"{results_dir}/class_imbal_byz{ratio}_fedavg.log"),
            'Krum': parse_log(f"{results_dir}/class_imbal_byz{ratio}_krum.log"),
            'Trimmed-Mean': parse_log(f"{results_dir}/class_imbal_byz{ratio}_tm.log"),
            'FLTG': parse_log(f"{results_dir}/class_imbal_byz{ratio}_fltg.log"),
        }

        byz_count = int(ratio * 20)
        print_comparison(
            f"Class Imbalance (2 classes/client) + {byz_count}/20 Byzantine ({ratio:.0%})",
            data,
            baseline
        )

    # ===== Experiment 3: Different Attacks =====
    print("\n\n🟢 MULTIPLE ATTACKS IN EXTREME CONDITIONS")
    print("Testing in worst-case: Ultra Extreme Non-IID (α=0.01) + 50% Byzantine\n")

    for attack in ['rop', 'ipm']:
        data = {
            'FedAVG': parse_log(f"{results_dir}/extreme_{attack}_fedavg.log"),
            'Krum': parse_log(f"{results_dir}/extreme_{attack}_krum.log"),
            'Trimmed-Mean': parse_log(f"{results_dir}/extreme_{attack}_tm.log"),
            'FLTG': parse_log(f"{results_dir}/extreme_{attack}_fltg.log"),
        }

        attack_name = "ROP" if attack == 'rop' else "IPM"
        print_comparison(
            f"{attack_name} Attack (α=0.01, 50% Byzantine)",
            data,
            baseline
        )

    # ===== SUMMARY =====
    print("\n\n" + "=" * 90)
    print(" 📊 SUMMARY & ANALYSIS")
    print("=" * 90)

    # Count wins across all extreme scenarios
    wins = {'FedAVG': 0, 'Krum': 0, 'Trimmed-Mean': 0, 'FLTG': 0}
    scenarios = []

    # Collect all scenarios
    test_scenarios = [
        ("Ultra Extreme 30%", "ultra_extreme_byz0.3"),
        ("Ultra Extreme 50%", "ultra_extreme_byz0.5"),
        ("Extreme 30%", "extreme_byz0.3"),
        ("Extreme 50%", "extreme_byz0.5"),
        ("Class Imbal 30%", "class_imbal_byz0.3"),
        ("Class Imbal 50%", "class_imbal_byz0.5"),
        ("Extreme ROP", "extreme_rop"),
        ("Extreme IPM", "extreme_ipm"),
    ]

    for scenario_name, prefix in test_scenarios:
        results = {}
        for method in ['fedavg', 'krum', 'tm', 'fltg']:
            _, acc = parse_log(f"{results_dir}/{prefix}_{method}.log")
            method_name = 'FedAVG' if method == 'fedavg' else 'Krum' if method == 'krum' else 'Trimmed-Mean' if method == 'tm' else 'FLTG'
            if acc:
                results[method_name] = acc

        if results:
            winner = max(results, key=results.get)
            wins[winner] += 1
            scenarios.append((scenario_name, winner, results[winner]))

    print("\n🏆 Win Count (Best accuracy in each scenario):")
    sorted_wins = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    for method, count in sorted_wins:
        percentage = count * 100 / len(scenarios) if scenarios else 0
        bar = "█" * count
        print(f"  {method:<15} {count:>2}/{len(scenarios)} wins ({percentage:>5.1f}%)  {bar}")

    print("\n📋 Scenario-by-Scenario Winners:")
    for scenario, winner, acc in scenarios:
        print(f"  {scenario:<20} → {winner:<15} ({acc:.2f}%)")

    print("\n💡 Key Insights:")

    # Check FLTG performance
    fltg_wins = wins['FLTG']
    total = len(scenarios)

    if fltg_wins >= total * 0.6:  # 60%+ wins
        print("  ✅ FLTG dominates in extreme Non-IID scenarios!")
        print("     The Non-IID aware weighting mechanism works as intended.")
    elif fltg_wins >= total * 0.4:  # 40-60% wins
        print("  🟡 FLTG shows competitive performance in extreme scenarios")
        print("     Mixed results - effective in some conditions but not universal")
    else:  # <40% wins
        print("  ❌ FLTG does NOT show advantage even in extreme Non-IID scenarios")
        print("     This contradicts the paper's core claim about Non-IID robustness")

    # Compare with simple FedAVG
    if wins['FedAVG'] > wins['FLTG']:
        diff = wins['FedAVG'] - wins['FLTG']
        print(f"\n  ⚠️  Even simple FedAVG outperforms FLTG by {diff} scenarios!")
        print("     Question: Is the complexity of FLTG justified?")

    # Check convergence speed
    print("\n📈 Convergence Analysis:")
    print("  🟢 Green: >10%p improvement (good convergence)")
    print("  🟡 Yellow: 5-10%p improvement (moderate)")
    print("  🔴 Red: <5%p improvement (slow convergence)")

    # Final verdict
    print("\n\n" + "=" * 90)
    print(" 🎯 FINAL VERDICT")
    print("=" * 90 + "\n")

    if fltg_wins >= total * 0.6:
        print("✅ Paper claims VALIDATED in extreme scenarios")
        print("   FLTG's Non-IID aware weighting provides clear advantage")
        print("   Recommendation: Use FLTG for highly heterogeneous data\n")
    elif fltg_wins >= total * 0.4:
        print("🟡 Paper claims PARTIALLY validated")
        print("   FLTG works in some extreme scenarios but not consistently")
        print("   Recommendation: Further tuning needed, use case dependent\n")
    else:
        print("❌ Paper claims NOT validated even in extreme scenarios")
        print("   Possible reasons:")
        print("   1. Implementation doesn't match paper's intent")
        print("   2. Dynamic reference selection picks wrong clients")
        print("   3. Paper's results may be dataset/model specific")
        print("   4. Hyperparameters (root dataset size, etc.) need tuning")
        print("\n   Recommendation: Contact paper authors for clarification\n")

    # Save results
    summary_file = "EXTREME_RESULTS_SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write("FLTG Extreme Non-IID Experimental Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Scenarios: {total}\n")
        f.write(f"FLTG Wins: {fltg_wins} ({fltg_wins*100/total:.1f}%)\n\n")
        f.write("Win Counts:\n")
        for method, count in sorted_wins:
            f.write(f"  {method}: {count}\n")
        f.write("\nConclusion: ")
        if fltg_wins >= total * 0.6:
            f.write("VALIDATED\n")
        elif fltg_wins >= total * 0.4:
            f.write("PARTIALLY VALIDATED\n")
        else:
            f.write("NOT VALIDATED\n")

    print(f"Summary saved to: {summary_file}\n")

if __name__ == "__main__":
    main()