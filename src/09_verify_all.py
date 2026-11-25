"""
Kiểm định tổng thể cuối cùng cho toàn bộ Cổng D1, E, F, Demo Streamlit và PDF.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
import torch
from pypdf import PdfReader

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "app"))

REQUIRED_FILES = [
    "KE_HOACH_TRIEN_KHAI_DT04.md",
    "README.md",
    "requirements.txt",
    ".gitignore",
    "data_card.md",
    "data/README.md",
    "data/data_manifest.csv",
    "data/data_schema.md",
    "data/sampling_log.csv",
    "data/external_events_reference.csv",
    "data/raw/gdelt_probe_log.json",
    "data/processed/corpus_monthly_totals.csv",
    "data/processed/cleaning_audit.json",
    "data/annotations/annotation_guideline.md",
    "data/annotations/gold_labels.csv",
    "data/annotations/gold_labels_split.csv",
    "data/annotations/review_log.csv",
    "data/annotations/human_audit_signoff.csv",
    "data/annotations/human_audit_summary.json",
    "data/splits/train_ids.csv",
    "data/splits/dev_ids.csv",
    "data/splits/test_ids.csv",
    "data/splits/external_uvn1_ids.csv",
    "configs/E0.json",
    "configs/E0_tfidf.json",
    "configs/E1.json",
    "configs/E2.json",
    "configs/E3.json",
    "evaluation/evaluation_protocol.md",
    "evaluation/experiments.csv",
    "evaluation/oof_cv_summary.csv",
    "evaluation/metrics_summary.csv",
    "evaluation/metrics_by_source_year.csv",
    "evaluation/metrics_by_boundary.csv",
    "evaluation/metrics_by_uncertainty_type.csv",
    "evaluation/e2_seeds_summary.csv",
    "evaluation/test_predictions_E0.csv",
    "evaluation/test_predictions_E0_tfidf.csv",
    "evaluation/test_predictions_E1.csv",
    "evaluation/test_predictions_E2.csv",
    "evaluation/test_predictions_E3.csv",
    "evaluation/error_analysis.md",
    "evaluation/extension_robustness.csv",
    "evaluation/extension_monthly_index.csv",
    "evaluation/extension_monthly_by_source.csv",
    "evaluation/extension_event_comparison.csv",
    "evaluation/extension_top_evidence_samples.csv",
    "evaluation/extension_timeline_stats.json",
    "reports/data_quality_report.md",
    "reports/Bao_cao_BTL.pdf",
    "reports/Nhat_ky_su_dung_AI.md",
    "reports/Tuyen_bo_su_dung_AI.md",
    "reports/figures/fig1_data_distribution.png",
    "reports/figures/fig2_model_comparison.png",
    "reports/figures/fig3_confusion_matrices.png",
    "reports/figures/fig4_cross_source_robustness.png",
    "reports/figures/fig5_raw_vs_normalized_timeline.png",
    "app/streamlit_app.py",
]


def main() -> None:
    print("1. Kiểm tra tệp bắt buộc...")
    missing = []
    empty = []
    for rel in REQUIRED_FILES:
        fp = ROOT_DIR / rel
        if not fp.exists():
            missing.append(rel)
        elif fp.stat().st_size == 0:
            empty.append(rel)
    assert not missing, f"Thiếu tệp: {missing}"
    assert not empty, f"Tệp rỗng: {empty}"
    print(f"  -> Đủ {len(REQUIRED_FILES)}/{len(REQUIRED_FILES)} tệp bắt buộc, không có tệp rỗng.")

    print("2. Kiểm tra rà soát bảo mật (không lộ API key) & kiểm định PDF...")
    secret_re = re.compile(r"(?:sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{20,})")
    for p in ROOT_DIR.rglob("*"):
        if any(part in p.parts for part in ("BTL_NPL_TQ", "__pycache__", "checkpoints", "models_cache")):
            continue
        if p.is_file() and p.suffix in {".py", ".md", ".json", ".csv", ".txt"}:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            assert not secret_re.search(txt), f"Phát hiện chuỗi giống API key trong {p}"

    reader = PdfReader(str(ROOT_DIR / "reports" / "Bao_cao_BTL.pdf"))
    pdf_txt = "\n".join(pg.extract_text() or "" for pg in reader.pages)
    pdf_compact = re.sub(r"\s+", "", pdf_txt)
    assert len(reader.pages) >= 4 and "BẤTĐỊNHKINHTẾ" in pdf_compact
    for token in ("1771040011", "1771040020", "1771040025", "LêThịThuỳTrang"):
        assert token in pdf_compact, f"Thiếu thông tin {token} trong PDF"
    print(f"  -> Security scan OK | PDF {len(reader.pages)} trang đọc Unicode tiếng Việt chuẩn xác.")

    print("3. Kiểm tra model inference trên GPU/CPU trong app/streamlit_app.py...")
    import streamlit_app  # noqa: E402
    from models_e0 import BBDVNRuleModel, TfidfBaselineModel  # noqa: E402
    from models_e1 import VietnameseBiEncoderProbe  # noqa: E402
    from models_e2 import SalienceDefinitionNLI  # noqa: E402
    from models_e3 import HybridStackingEnsemble  # noqa: E402

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sample_key = list(streamlit_app.SAMPLE_INPUTS.keys())[0]
    sample_txt = streamlit_app.SAMPLE_INPUTS[sample_key]
    sample_ttl = streamlit_app.SAMPLE_TITLES.get(sample_key)
    e0 = BBDVNRuleModel(ROOT_DIR / "configs" / "E0.json")
    e0t = TfidfBaselineModel.load(ROOT_DIR / "checkpoints" / "e0_tfidf.joblib")
    e1 = VietnameseBiEncoderProbe.load(ROOT_DIR / "checkpoints" / "e1_model.joblib", device=dev)
    e2 = SalienceDefinitionNLI.load(ROOT_DIR / "checkpoints" / "e2_best", device=dev)
    e3 = HybridStackingEnsemble.load(ROOT_DIR / "checkpoints" / "e3_stacking.joblib")
    r0 = e0.predict_one(sample_txt, title=sample_ttl)
    r0t = e0t.predict_batch([sample_txt], titles=[sample_ttl], return_sent_scores=True)[0]
    r1 = e1.predict_batch([sample_txt], titles=[sample_ttl], return_sent_scores=True)[0]
    r2 = e2.predict_batch([sample_txt], titles=[sample_ttl], use_ensemble=True, return_sent_scores=True)[0]
    r3 = e3.predict_from_base_preds([r0], [r0t], [r1], [r2], texts=[sample_txt], titles=[sample_ttl])[0]
    print(
        f"  -> Demo models smoke inference ({dev}): "
        f"E0={r0['pred_label']}, E0_tfidf={r0t['pred_label']}, E1={r1['pred_label']}, "
        f"E2={r2['pred_label']}, E3={r3['pred_label']} ({r3['pred_uncertainty_type']})"
    )
    del e1, e2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("4. Smoke test khởi chạy Streamlit HTTP server...")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT_DIR / "app" / "streamlit_app.py"),
            "--server.headless",
            "true",
            "--server.port",
            "8509",
            "--browser.gatherUsageStats",
            "false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(ROOT_DIR),
    )
    try:
        ok = False
        for _ in range(25):
            try:
                resp = requests.get("http://127.0.0.1:8509/_stcore/health", timeout=2)
                if resp.status_code == 200:
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(1)
        assert ok, "Streamlit server không trả về HTTP 200 trong thời gian chờ!"
        page = requests.get("http://127.0.0.1:8509/", timeout=5)
        assert page.status_code == 200
        print("  -> Streamlit HTTP /_stcore/health=200 & /=200 OK!")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("5. Kiểm chứng độc lập evaluation/experiments.csv bằng evaluate_predictions.py --verify...")
    res = subprocess.run(
        [sys.executable, str(ROOT_DIR / "src" / "evaluate_predictions.py"), "--verify"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    print(res.stdout)
    print("=== FINAL VERIFICATION PASSED 100% ===")


if __name__ == "__main__":
    main()
