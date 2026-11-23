#!/usr/bin/env python3
"""Build and clean-copy verify research_code_v6.zip and research_manuscript.pdf."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
DELIV_DIR = ROOT / "deliverables/research_submission"
CODE_ZIP = DELIV_DIR / "research_code_v6.zip"
MANUSCRIPT_MD = DELIV_DIR / "manuscript.md"
MANUSCRIPT_PDF = DELIV_DIR / "research_manuscript.pdf"
VERIF_JSON = DELIV_DIR / "SUBMISSION_VERIFICATION.json"
MAX_BYTES = 10_000_000

CORE_SCRIPTS = [
    "analyze_spectral_shape_mismatch_v1.py",
    "assemble_acoustic_research_checkpoint_v1.py",
    "audit_experiment007_decision_gate_v1.py",
    "audit_real_acoustic_development_v1.py",
    "audit_simulation_realism_v1.py",
    "build_experiment004_frozen_mixtures.py",
    "build_real_acoustic_results_v2.py",
    "build_site_context_figure.py",
    "build_synthetic_mixtures_v1.py",
    "build_synthetic_mixtures_v2.py",
    "run_experiment006_contrast_robustness_v1.py",
    "run_experiment007_source_disjoint_v1.py",
    "summarize_real_acoustic_development_v1.py",
    "train_synthetic_baselines_v1.py",
    "train_synthetic_baselines_v2.py",
    "verify_experiment007_artifacts_v1.py",
    "verify_real_acoustic_results_v2.py",
]

ZIP_README = """# Underwater Acoustic Research Code (V6)

Canonical runnable research package (17 science scripts + site-context summary).
Competition data, VTUAD WAV audio, and saved assessment predictions are NOT included.

## 1. Environment
Install pinned dependencies from `requirements.txt` (tested on Python 3.13).

## 2. Site-Context Figure (Figure 4)
```bash
python scripts/build_site_context_figure.py \\
  --archive /PATH/TO/South_African_Real_Acoustics.zip \\
  --summary site_context/training_only_site_aggregates.csv \\
  --output site_context_candidate_locations.png
```

## 3. Experiment 007 Decision-Gate Audit (Table 2)
```bash
python scripts/audit_experiment007_decision_gate_v1.py --self-test-only
python scripts/audit_experiment007_decision_gate_v1.py \\
  --input /PATH/TO/reports/baselines/experiment007_source_disjoint_v1 \\
  --output ./exp007_gate_out
```

## 4. Six-Station Descriptive Appendix (Table 1 & Figures 1-3, 5-6)
Requires `reports/audits/real_acoustic_development_v1/` and
`reports/research_manuscript_v1/station_relative_anomaly_proxy.csv`:
```bash
python scripts/build_real_acoustic_results_v2.py --root . --output ./real_acoustic_out
python scripts/verify_real_acoustic_results_v2.py --root . --output ./real_acoustic_out
```

## Limits
Archive levels are uncalibrated source-export dB (not dB re 1 uPa). Coordinates in
`stations.csv` are working site candidates without confirmed vessel-event labels.
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_code_zip() -> dict:
    """Create research_code_v6.zip and enforce strict content/size checks."""
    DELIV_DIR.mkdir(parents=True, exist_ok=True)
    agg_csv = DELIV_DIR / "training_only_site_aggregates.csv"
    req_txt = ROOT / "requirements.txt"
    assert agg_csv.is_file(), f"Missing {agg_csv}"
    assert req_txt.is_file(), f"Missing {req_txt}"

    tmp_zip = CODE_ZIP.with_suffix(".tmp")
    with ZipFile(tmp_zip, "w", ZIP_DEFLATED) as z:
        z.writestr("README.md", ZIP_README)
        z.write(req_txt, "requirements.txt")
        z.write(agg_csv, "site_context/training_only_site_aggregates.csv")
        for name in CORE_SCRIPTS:
            sp = ROOT / "scripts" / name
            assert sp.is_file(), f"Missing script: {sp}"
            z.write(sp, f"scripts/{name}")
    tmp_zip.replace(CODE_ZIP)

    banned = ("assessment_predictions.csv", "calibration_predictions.csv", ".parquet", ".wav", "test.csv", "mmsi")
    with ZipFile(CODE_ZIP) as z:
        assert z.testzip() is None, "ZIP CRC check failed"
        members = [i.filename for i in z.infolist() if not i.is_dir()]
        assert len(members) == len(set(members)) == 20
        for m in members:
            assert not any(b in m.casefold() for b in banned), f"Banned archive member: {m}"

    size = CODE_ZIP.stat().st_size
    assert size < MAX_BYTES, f"ZIP exceeds 10 MB: {size}"
    return {"path": str(CODE_ZIP.relative_to(ROOT)), "bytes": size, "sha256": sha256_file(CODE_ZIP), "entries": 20}


def verify_code_zip_clean_copy() -> dict:
    """Extract research_code_v6.zip to a temp dir and verify compilation + clean-copy runs."""
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    py = sys.executable

    with tempfile.TemporaryDirectory(prefix="v6_clean_test_") as td:
        tmp = Path(td)
        with ZipFile(CODE_ZIP) as z:
            z.extractall(tmp)

        # 1. py_compile all 17 scripts
        for name in CORE_SCRIPTS:
            py_compile.compile(str(tmp / "scripts" / name), doraise=True)

        # 2. Site-context figure script
        fig_out = tmp / "site_context_test.png"
        subprocess.run(
            [
                py,
                str(tmp / "scripts/build_site_context_figure.py"),
                "--archive",
                str(ROOT / "data/zindi/raw/South_African_Real_Acoustics.zip"),
                "--summary",
                str(tmp / "site_context/training_only_site_aggregates.csv"),
                "--output",
                str(fig_out),
            ],
            check=True,
            env=env,
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        assert fig_out.is_file() and fig_out.stat().st_size > 10_000

        # 3. Experiment 007 decision-gate self-test & byte-identical CSV reproduction
        gate_script = str(tmp / "scripts/audit_experiment007_decision_gate_v1.py")
        subprocess.run([py, gate_script, "--self-test-only"], check=True, env=env, cwd=tmp, capture_output=True)
        gate_out = tmp / "exp007_gate_out"
        subprocess.run(
            [
                py,
                gate_script,
                "--input",
                str(ROOT / "reports/baselines/experiment007_source_disjoint_v1"),
                "--output",
                str(gate_out),
            ],
            check=True,
            env=env,
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        ref_gate = ROOT / "reports/audits/experiment007_decision_gate_v1"
        gate_csvs = ["policy_comparison.csv", "calibration_assessment_gap.csv", "source_deletion_summary.csv"]
        for csv_name in gate_csvs:
            assert (gate_out / csv_name).read_bytes() == (ref_gate / csv_name).read_bytes(), f"Mismatch: {csv_name}"

        # 4. Real-acoustic descriptive v2 build + verify
        (tmp / "reports/audits").mkdir(parents=True)
        (tmp / "reports/research_manuscript_v1").mkdir(parents=True)
        os.symlink(
            ROOT / "reports/audits/real_acoustic_development_v1",
            tmp / "reports/audits/real_acoustic_development_v1",
        )
        os.symlink(
            ROOT / "reports/research_manuscript_v1/station_relative_anomaly_proxy.csv",
            tmp / "reports/research_manuscript_v1/station_relative_anomaly_proxy.csv",
        )
        ra_out = tmp / "real_acoustic_out"
        subprocess.run(
            [py, str(tmp / "scripts/build_real_acoustic_results_v2.py"), "--root", str(tmp), "--output", str(ra_out)],
            check=True,
            env=env,
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [py, str(tmp / "scripts/verify_real_acoustic_results_v2.py"), "--root", str(tmp), "--output", str(ra_out)],
            check=True,
            env=env,
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        ref_ra = ROOT / "reports/research_manuscript_v1/real_acoustic_results_v2"
        for csv_name in ["validated_station_summary.csv", "run_bootstrap_intervals.csv", "robust_z_availability.csv"]:
            assert (ra_out / csv_name).read_bytes() == (ref_ra / csv_name).read_bytes(), f"RA mismatch: {csv_name}"

    return {
        "py_compile_all_17": "PASS",
        "site_context_figure_regen": "PASS",
        "exp007_decision_gate_byte_identical_csvs": gate_csvs,
        "real_acoustic_results_v2_byte_identical_csvs": [
            "validated_station_summary.csv",
            "run_bootstrap_intervals.csv",
            "robust_z_availability.csv",
        ],
    }


def _md_inline_to_rl(text: str) -> str:
    """Convert minimal inline markdown (**bold**, *italic*, `code`) into ReportLab XML."""
    s = escape(text)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+)`", r"<font face='Courier' size='8.5'>\1</font>", s)
    return s


def _add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawRightString(LETTER[0] - 0.65 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.drawString(0.65 * inch, 0.45 * inch, "French Embassy Underwater Vessel Noise Challenge — Research Write-Up")
    canvas.restoreState()


def render_manuscript_pdf(md_text: str) -> dict:
    """Render manuscript.md and its 6 figures into a concise, print-legible PDF."""
    styles = getSampleStyleSheet()
    title_st = ParagraphStyle("TitleSt", parent=styles["Heading1"], fontSize=14, leading=17, spaceAfter=6, textColor=colors.HexColor("#112233"))
    h2_st = ParagraphStyle("H2St", parent=styles["Heading2"], fontSize=11, leading=14, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#1a365d"), keepWithNext=True)
    h3_st = ParagraphStyle("H3St", parent=styles["Heading3"], fontSize=9.8, leading=12.5, spaceBefore=6, spaceAfter=2, textColor=colors.HexColor("#224466"), keepWithNext=True)
    body_st = ParagraphStyle("BodySt", parent=styles["Normal"], fontSize=9.0, leading=12.2, spaceAfter=4)
    bullet_st = ParagraphStyle("BulletSt", parent=body_st, leftIndent=14, firstLineIndent=-8, spaceAfter=2.5)
    cap_st = ParagraphStyle("CapSt", parent=styles["Normal"], fontSize=8.0, leading=10.5, textColor=colors.HexColor("#333333"), spaceAfter=6)
    cell_st = ParagraphStyle("CellSt", parent=styles["Normal"], fontSize=8.0, leading=10.0)
    hdr_st = ParagraphStyle("HdrSt", parent=cell_st, textColor=colors.white, fontName="Helvetica-Bold")

    story = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(_md_inline_to_rl(line[2:].strip()), title_st))
            i += 1
        elif line.startswith("## "):
            story.append(Paragraph(_md_inline_to_rl(line[3:].strip()), h2_st))
            i += 1
        elif line.startswith("### "):
            story.append(Paragraph(_md_inline_to_rl(line[4:].strip()), h3_st))
            i += 1
        elif line.startswith("!["):
            m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", line)
            if m:
                fig_path = DELIV_DIR / m.group(1)
                assert fig_path.is_file(), f"Missing figure: {fig_path}"
                from PIL import Image as PILImage
                with PILImage.open(fig_path) as im:
                    w_px, h_px = im.size
                max_w = 6.7 * inch
                max_h = 2.65 * inch
                scale = min(max_w / w_px, max_h / h_px)
                story.append(RLImage(str(fig_path), width=w_px * scale, height=h_px * scale))
                story.append(Spacer(1, 2))
            i += 1
        elif line.startswith("|"):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i].strip())
                i += 1
            rows_raw = []
            for tl in tbl_lines:
                parts = [c.strip() for c in tl.strip("|").split("|")]
                if all(re.fullmatch(r":?-+:?", c) for c in parts):
                    continue
                rows_raw.append(parts)
            if rows_raw:
                data = []
                for r_idx, r in enumerate(rows_raw):
                    st = hdr_st if r_idx == 0 else cell_st
                    data.append([Paragraph(_md_inline_to_rl(c), st) for c in r])
                ncols = len(data[0])
                avail_w = 7.2 * inch
                col_w = [avail_w * 0.34] + [avail_w * 0.66 / (ncols - 1)] * (ncols - 1) if ncols > 1 else [avail_w]
                tbl = Table(data, colWidths=col_w, repeatRows=1)
                tbl.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a365d")),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fa")]),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                            ("LEFTPADDING", (0, 0), (-1, -1), 5),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ]
                    )
                )
                story.append(tbl)
                story.append(Spacer(1, 5))
        elif line.startswith(("**Figure ", "**Table ")):
            story.append(Paragraph(_md_inline_to_rl(line), cap_st))
            i += 1
        elif re.match(r"^(\d+\.|-)\s+", line):
            para_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("#", "|", "![", "-", "1.", "2.", "3.", "4.", "5.", "6.")):
                para_lines.append(lines[i].strip())
                i += 1
            story.append(Paragraph(_md_inline_to_rl(" ".join(para_lines)), bullet_st))
        else:
            para_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("#", "|", "![", "- ")) and not re.match(r"^\d+\.\s+", lines[i].strip()):
                para_lines.append(lines[i].strip())
                i += 1
            story.append(Paragraph(_md_inline_to_rl(" ".join(para_lines)), body_st))

    tmp_pdf = MANUSCRIPT_PDF.with_suffix(".tmp.pdf")
    doc = SimpleDocTemplate(
        str(tmp_pdf),
        pagesize=LETTER,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.6 * inch,
        title="South African Underwater Acoustic Variability & Synthetic Vessel-Energy Limits",
        author="Competition Research Team",
    )
    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    tmp_pdf.replace(MANUSCRIPT_PDF)

    reader = PdfReader(str(MANUSCRIPT_PDF))
    pages = len(reader.pages)
    extracted = " ".join((p.extract_text() or "") for p in reader.pages)
    pdf_bytes = MANUSCRIPT_PDF.stat().st_size
    assert MANUSCRIPT_PDF.read_bytes().startswith(b"%PDF-")
    assert pdf_bytes < MAX_BYTES, f"PDF exceeds 10 MB: {pdf_bytes}"
    assert 4 <= pages <= 10, f"PDF page count {pages} outside Zindi 4-10 page limit"
    assert "relabelling Figure 6" not in extracted
    assert "Figure 4" in extracted
    return {
        "path": str(MANUSCRIPT_PDF.relative_to(ROOT)),
        "bytes": pdf_bytes,
        "pages": pages,
        "sha256": sha256_file(MANUSCRIPT_PDF),
        "figure4_fix_verified": True,
    }


def verify_existing_deliverables() -> dict[str, object]:
    """Verify existing research_code_v6.zip, PDF, and metadata without rewriting them."""
    assert VERIF_JSON.is_file(), f"Missing {VERIF_JSON}"
    assert CODE_ZIP.is_file(), f"Missing {CODE_ZIP}"
    assert MANUSCRIPT_PDF.is_file(), f"Missing {MANUSCRIPT_PDF}"

    recorded = json.loads(VERIF_JSON.read_text(encoding="utf-8"))
    code_bytes = CODE_ZIP.stat().st_size
    code_sha = sha256_file(CODE_ZIP)
    assert code_bytes < MAX_BYTES, f"Zip exceeds 10 MB: {code_bytes}"
    assert code_bytes == recorded["code_archive"]["bytes"], "Zip byte count mismatch"
    assert code_sha == recorded["code_archive"]["sha256"], "Zip SHA-256 mismatch"

    pdf_bytes = MANUSCRIPT_PDF.stat().st_size
    pdf_sha = sha256_file(MANUSCRIPT_PDF)
    reader = PdfReader(str(MANUSCRIPT_PDF))
    pages = len(reader.pages)
    extracted = " ".join((p.extract_text() or "") for p in reader.pages)
    assert MANUSCRIPT_PDF.read_bytes().startswith(b"%PDF-")
    assert pdf_bytes < MAX_BYTES, f"PDF exceeds 10 MB: {pdf_bytes}"
    assert 4 <= pages <= 10, f"PDF page count {pages} outside Zindi 4-10 page limit"
    assert "Figure 4" in extracted and "relabelling Figure 6" not in extracted
    assert pdf_bytes == recorded["research_pdf"]["bytes"], "PDF byte count mismatch"
    assert pdf_sha == recorded["research_pdf"]["sha256"], "PDF SHA-256 mismatch"

    clean_checks = verify_code_zip_clean_copy()
    return {
        "status": recorded["status"],
        "verify_only": True,
        "code_archive": {"path": str(CODE_ZIP.relative_to(ROOT)), "bytes": code_bytes, "sha256": code_sha},
        "clean_copy_verification": clean_checks,
        "research_pdf": {"path": str(MANUSCRIPT_PDF.relative_to(ROOT)), "bytes": pdf_bytes, "pages": pages, "sha256": pdf_sha},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify Zindi research submission deliverables.")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing deliverables without rebuilding.")
    args = parser.parse_args()

    if args.verify_only:
        print(json.dumps(verify_existing_deliverables(), indent=2))
        return

    code_info = build_code_zip()
    clean_checks = verify_code_zip_clean_copy()

    raw_md = MANUSCRIPT_MD.read_text(encoding="utf-8")
    updated_md = raw_md
    updated_md = re.sub(
        r"`(\{\{CODE_ZIP_BYTES\}\}|[0-9,]+)` bytes, SHA-256 `(\{\{CODE_ZIP_SHA256\}\}|[0-9a-f]{64})`",
        f"`{code_info['bytes']}` bytes, SHA-256 `{code_info['sha256']}`",
        updated_md,
    )
    if updated_md != raw_md:
        MANUSCRIPT_MD.write_text(updated_md, encoding="utf-8")

    pdf_info = render_manuscript_pdf(updated_md)
    report = {
        "status": "VERIFIED_LOCAL_CANDIDATES_HOLD_FOR_ZINDI_ACTIONS",
        "code_archive": code_info,
        "clean_copy_verification": clean_checks,
        "research_pdf": pdf_info,
    }
    VERIF_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
