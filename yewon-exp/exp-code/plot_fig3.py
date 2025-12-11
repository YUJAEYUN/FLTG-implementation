import pandas as pd
import matplotlib.pyplot as plt

# 결과 CSV 파일 경로
CSV_PATH = "results_fig3_cifar10_client100_simplecifar_epoch5.csv"

def main():
    # CSV 로드
    df = pd.read_csv(CSV_PATH)

    # test_acc가 0~1 스케일이라고 가정해서 % 변환
    df["test_acc_pct"] = df["test_acc"] * 100.0

    # bias_prob 리스트 자동 추출
    bias_values = sorted(df["bias_prob"].unique())

    # 서브플롯 생성 (Fig.3은 bias_prob 3개 → 1x3)
    n_bias = len(bias_values)
    fig, axes = plt.subplots(
        1, n_bias, figsize=(5 * n_bias, 4),
        sharey=True
    )

    if n_bias == 1:
        axes = [axes]

    for ax, bias in zip(axes, bias_values):

        # bias 별 데이터 필터링
        sub = df[df["bias_prob"] == bias]
        attacked = sub[sub["traitor"] > 0]     # 공격 구간만

        # FLTG / FLTrust 정렬
        fltg = attacked[attacked["aggr"] == "fltg"].sort_values("traitor")
        fl_trust = attacked[attacked["aggr"] == "fl_trust"].sort_values("traitor")

        # no-attack baseline
        fedavg_row = sub[
            (sub["traitor"] == 0.0)
            & (sub["aggr"] == "avg_no_attack")
        ]

        # --- 플롯: FLTG ---
        if not fltg.empty:
            ax.plot(
                fltg["traitor"],
                fltg["test_acc_pct"],
                marker="o", markersize=5,
                linewidth=2,
                label="FLTG"
            )

        # --- 플롯: FLTrust ---
        if not fl_trust.empty:
            ax.plot(
                fl_trust["traitor"],
                fl_trust["test_acc_pct"],
                marker="s", markersize=5,
                linewidth=2,
                label="FLTrust"
            )

        # --- FedAvg Baseline (점선) ---
        if not fedavg_row.empty:
            baseline = float(fedavg_row["test_acc_pct"].iloc[0])
            ax.axhline(
                y=baseline,
                linestyle="--",
                linewidth=2,
                color="tab:green",
                label="FedAvg (no attack)"
            )

        # --- 그래프 제목/축 설정 ---
        ax.set_title(f"bias_prob = {bias}", fontsize=12)
        ax.set_xlabel("Malicious client ratio (%)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Test Accuracy (%)", fontsize=11)

    # 전역 Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=3,
        fontsize=11
    )

    fig.suptitle("Fig.3-style Reproduction of FLTG", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()