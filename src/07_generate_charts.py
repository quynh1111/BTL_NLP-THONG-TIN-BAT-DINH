"""
Bước 7: Sinh 5 biểu đồ độ phân giải cao (300 DPI) cho Báo cáo PDF và Demo.
Toàn bộ số liệu đọc trực tiếp từ evaluation/*.csv và data/processed/*.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT_DIR / "data" / "processed"
ANN_DIR = ROOT_DIR / "data" / "annotations"
DATA_DIR = ROOT_DIR / "data"
EVAL_DIR = ROOT_DIR / "evaluation"
FIG_DIR = ROOT_DIR / "reports" / "figures"

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Arial")


def chart1_data_distribution() -> None:
    totals = pd.read_csv(PROC_DIR / "corpus_monthly_totals.csv", encoding="utf-8-sig")
    totals["year"] = totals["year_month"].str.slice(0, 4)
    totals = totals[totals["year"].isin(["2023", "2024", "2025"])]
    yr_src = totals.groupby(["year", "source"])["total_econ_news"].sum().unstack(fill_value=0)

    gold = pd.read_csv(ANN_DIR / "gold_labels_split.csv", encoding="utf-8-sig")
    von = gold[gold["dataset_source"] == "VietOnlineNews"]
    bt_ct = pd.crosstab(von["boundary_type"], von["task_label"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    yr_src.plot(kind="bar", ax=axes[0], color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="black", linewidth=0.5)
    axes[0].set_title("Sản lượng bài kinh tế VietOnlineNews theo năm & báo (N = 39.878)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Năm công bố")
    axes[0].set_ylabel("Số bài báo (Kinh doanh + BĐS)")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="Nguồn báo")
    for container in axes[0].containers:
        axes[0].bar_label(container, fontsize=8, padding=2)

    bt_labels_vi = {
        "clear_uncertain": "Bất định tri thức rõ ràng",
        "implicit_uncertain": "Bất định tri thức ẩn ý",
        "aleatoric_volatility": "Bất định ngẫu nhiên\n(biến động giá/TT)",
        "negative_certain_hard_negative": "Tin xấu chắc chắn\n(Hard-Negative)",
        "neutral_factual": "Trung tính / số liệu",
        "pr_or_positive": "Tích cực / PR",
    }
    bt_ct.index = [bt_labels_vi.get(i, i) for i in bt_ct.index]
    bt_ct.plot(kind="barh", stacked=True, ax=axes[1], color=["#4c72b0", "#dd8452"], edgecolor="black", linewidth=0.5)
    axes[1].set_title(f"Phân bố tập Gold chính theo nhóm biên giới (n = {len(von)})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Số đoạn văn bản")
    axes[1].set_ylabel("")
    axes[1].legend(title="Nhãn Gold")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_data_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def chart2_model_comparison() -> None:
    ms = pd.read_csv(EVAL_DIR / "metrics_summary.csv", encoding="utf-8-sig")
    te = ms[ms["split"] == "test"].copy()
    dv = ms[ms["split"] == "dev"].copy()

    order = ["E0", "E0_tfidf", "E1", "E2", "E3"]
    order = [r for r in order if r in set(te["run_id"].astype(str))]
    te["run_id"] = pd.Categorical(te["run_id"], categories=order, ordered=True)
    te = te.sort_values("run_id")
    dv["run_id"] = pd.Categorical(dv["run_id"], categories=order, ordered=True)
    dv = dv.sort_values("run_id")

    x = np.arange(len(te))
    w = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    b1 = axes[0].bar(x - w, dv["macro_f1"], width=w, label="Dev OOF Macro-F1", color="#8da0cb", edgecolor="black", linewidth=0.5)
    b2 = axes[0].bar(x, te["macro_f1"], width=w, label="Test Macro-F1", color="#fc8d62", edgecolor="black", linewidth=0.5)
    b3 = axes[0].bar(x + w, te["f1_uncertain"], width=w, label="Test F1 (uncertain)", color="#66c2a5", edgecolor="black", linewidth=0.5)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(te["run_id"])
    axes[0].set_ylim(0, 1.18)
    axes[0].set_title("So sánh hiệu năng Dev OOF vs Test độc lập (E0 → E3)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Điểm F1")
    axes[0].legend(loc="upper left", fontsize=8.5, ncol=1)
    for b in [b1, b2, b3]:
        axes[0].bar_label(b, fmt="%.3f", fontsize=7.5, padding=2)

    # Biểu đồ 2: Hard-Negative FPR và Evidence Token-F1
    b4 = axes[1].bar(x - 0.17, te["hard_neg_fpr"], width=0.34, label="Hard-Negative FPR (thấp hơn = tốt hơn)", color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)
    b5 = axes[1].bar(x + 0.17, te["evidence_token_f1_uncertain"], width=0.34, label="Evidence Token-F1 (cao hơn = tốt hơn)", color="#2ecc71", alpha=0.85, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(te["run_id"])
    axes[1].set_ylim(0, 1.12)
    axes[1].set_title("Chống nhầm 'tin xấu chắc chắn' & Chất lượng câu chứng cứ", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Tỷ lệ / Token-F1")
    axes[1].legend(loc="upper right", fontsize=8.5)
    for b in [b4, b5]:
        axes[1].bar_label(b, fmt="%.3f", fontsize=8, padding=2)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def chart3_confusion_matrices() -> None:
    ms = pd.read_csv(EVAL_DIR / "metrics_summary.csv", encoding="utf-8-sig")
    te = ms[ms["split"] == "test"].set_index("run_id")
    runs = [r for r in ["E0", "E0_tfidf", "E1", "E2", "E3"] if r in te.index]

    fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 3.6))
    for ax, r in zip(axes, runs):
        row = te.loc[r]
        cm = np.array([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["not_unc", "uncertain"],
            yticklabels=["not_unc", "uncertain"],
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title(f"{r}\nMacro-F1={row['macro_f1']:.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Nhãn Gold" if r == runs[0] else "")

    plt.suptitle("Ma trận nhầm lẫn trên tập Test độc lập (n = 60)", fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()


def chart4_cross_source_robustness() -> None:
    rob = pd.read_csv(EVAL_DIR / "extension_robustness.csv", encoding="utf-8-sig")
    order = [r for r in ["E0", "E0_tfidf", "E1", "E2", "E3"] if r in set(rob["run_id"].astype(str))]
    rob["run_id"] = pd.Categorical(rob["run_id"], categories=order, ordered=True)
    piv = rob.pivot(index="target_source", columns="run_id", values="macro_f1")

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    piv.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black", linewidth=0.5)
    ax.set_title("Phần mở rộng 1: Độ bền xuyên nguồn (LOSO trên 3 báo & Zero-Transfer UVN-1 4 báo mới)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("Nguồn báo kiểm thử (bị loại khỏi tập huấn luyện)")
    ax.set_ylim(0, 1.18)
    ax.tick_params(axis="x", rotation=12)
    ax.legend(title="Cấu hình", loc="upper center", ncol=len(order), fontsize=8.5)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7.5, padding=2)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_cross_source_robustness.png", dpi=300, bbox_inches="tight")
    plt.close()


def chart5_raw_vs_normalized_timeline() -> None:
    idx_df = pd.read_csv(EVAL_DIR / "extension_monthly_index.csv", encoding="utf-8-sig")
    events_df = pd.read_csv(DATA_DIR / "external_events_reference.csv", encoding="utf-8-sig")
    best_model = str(idx_df["best_model_used"].iloc[0])

    x = np.arange(len(idx_df))
    fig, ax1 = plt.subplots(figsize=(13.5, 5.5))

    ax1.plot(
        x,
        idx_df["raw_count_scaled_100"],
        linestyle="--",
        marker="o",
        markersize=4,
        color="#95a5a6",
        linewidth=1.8,
        label=f"Tần suất THÔ quy đổi (mean=100) — bị lệch do sản lượng crawl tăng mạnh năm 2025",
    )
    ax1.plot(
        x,
        idx_df["bbd_index_best"],
        linestyle="-",
        marker="s",
        markersize=5,
        color="#c0392b",
        linewidth=2.4,
        label=f"Chỉ số CHUẨN HÓA BBD-style ({best_model}, mean=100) — chuẩn hóa theo tổng tin từng báo",
    )
    ax1.axhline(100.0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)

    # Tô vùng sự kiện tham chiếu với nhãn đặt gọn gàng ở phía trên tránh đè đường
    colors_ev = ["#3498db", "#e67e22", "#27ae60", "#8e44ad"]
    ym_to_x = {ym: i for i, ym in enumerate(idx_df["year_month"])}
    max_curve = float(idx_df[["raw_count_scaled_100", "bbd_index_best"]].max().max())
    ymax = max(225.0, max_curve + 45.0)
    y_offsets = [ymax * 0.74, ymax * 0.66, ymax * 0.59, ymax * 0.80]
    for i, (_, er) in enumerate(events_df.iterrows()):
        s_x = ym_to_x.get(er["start_month"], 0)
        e_x = ym_to_x.get(er["end_month"], s_x)
        ax1.axvspan(s_x - 0.3, e_x + 0.3, color=colors_ev[i % len(colors_ev)], alpha=0.12)
        peak_x = ym_to_x.get(er["year_month"], s_x)
        y_val = float(idx_df.loc[peak_x, "bbd_index_best"])
        y_box = max(y_offsets[i % len(y_offsets)], y_val + 14.0)
        y_box = min(y_box, ymax - 16.0)
        ax1.annotate(
            er["event_label"],
            xy=(peak_x, y_val),
            xytext=(peak_x, y_box),
            fontsize=8,
            fontweight="bold",
            ha="center",
            arrowprops=dict(arrowstyle="->", color=colors_ev[i % len(colors_ev)], lw=1.1),
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=colors_ev[i % len(colors_ev)], alpha=0.95),
        )

    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(idx_df["year_month"].iloc[::2], rotation=35, ha="right", fontsize=8.5)
    ax1.set_ylabel("Chỉ số tương đối (Trung bình 36 tháng = 100)")
    ax1.set_title(
        "Phần mở rộng 2 & 3: Tần suất bất định Thô vs Chuẩn hóa trên 3 báo kinh tế (2023–2025)\n"
        "[Cảnh báo: Chỉ là thống kê tần suất văn bản trên corpus khảo sát, KHÔNG phải chỉ số EPU chính thức hay dự báo vĩ mô]",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.set_ylim(25, ymax)
    ax1.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_raw_vs_normalized_timeline.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    chart1_data_distribution()
    chart2_model_comparison()
    chart3_confusion_matrices()
    chart4_cross_source_robustness()
    chart5_raw_vs_normalized_timeline()
    print("Generated 5 charts in:", FIG_DIR)


if __name__ == "__main__":
    main()
