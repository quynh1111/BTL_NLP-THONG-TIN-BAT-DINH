"""
Bước 8: Biên dịch Báo cáo Kỹ thuật PDF (reports/Bao_cao_BTL.pdf) từ nguồn XeLaTeX
trong reports/bao_cao_latex/main.tex (kế thừa chuẩn Template_BCTT_DATN).
Đồng bộ ảnh biểu đồ mới nhất từ reports/figures/ vào reports/bao_cao_latex/images/
và chạy xelatex 3 lượt để đảm bảo mục lục, danh mục bảng/sơ đồ/hình và trích dẫn khớp 100%.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"
FIG_DIR = REPORTS_DIR / "figures"
LATEX_DIR = REPORTS_DIR / "bao_cao_latex"
LATEX_IMG_DIR = LATEX_DIR / "images"
PDF_PATH = REPORTS_DIR / "Bao_cao_BTL.pdf"


def find_xelatex() -> str:
    exe = shutil.which("xelatex")
    if exe:
        return exe
    fallback = Path(r"C:\Program Files\MiKTeX\miktex\bin\x64\xelatex.exe")
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("Không tìm thấy trình biên dịch xelatex trên hệ thống.")


def sync_figures() -> None:
    LATEX_IMG_DIR.mkdir(parents=True, exist_ok=True)
    for fig in FIG_DIR.glob("fig*.png"):
        shutil.copy2(fig, LATEX_IMG_DIR / fig.name)
    logo_src = ROOT_DIR / "Template_BCTT_DATN" / "images" / "logo-dai-nam.png"
    logo_dst = LATEX_IMG_DIR / "logo-dai-nam.png"
    if logo_src.exists() and not logo_dst.exists():
        shutil.copy2(logo_src, logo_dst)


def main() -> None:
    main_tex = LATEX_DIR / "main.tex"
    if not main_tex.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp nguồn LaTeX: {main_tex}")

    sync_figures()
    xelatex_bin = find_xelatex()
    cmd = [xelatex_bin, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]

    for pass_idx in range(1, 4):
        res = subprocess.run(
            cmd,
            cwd=str(LATEX_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode != 0:
            tail = "\n".join(res.stdout.splitlines()[-40:])
            raise RuntimeError(f"XeLaTeX lỗi ở lượt {pass_idx}:\n{tail}")

    built_pdf = LATEX_DIR / "main.pdf"
    if not built_pdf.exists():
        raise FileNotFoundError(f"Không tạo được tệp {built_pdf}")
    shutil.copy2(built_pdf, PDF_PATH)
    print(f"Built XeLaTeX PDF report successfully: {PDF_PATH}")


if __name__ == "__main__":
    main()
