"""Convert ALICE Papers I/II/III to PDF — robust math handling.

Usage: python md2pdf.py
"""
import re
import subprocess
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).parent
PAPERS = [
    "Paper_I_Minimum_Reflection_Principle",
    "Paper_II_Body_Brain_Integration",
    "Paper_III_Emergent_Psychopathology",
]


def preprocess(md: str) -> str:
    """Fix markdown math for pandoc+xelatex compatibility."""

    # 1. \\\vert → | (triple-escaped pipe)
    md = md.replace("\\\\\\vert", "|")

    # 2. Fix space before closing $ in math: `\vert $` → `\vert$`
    #    Pandoc requires no space immediately before closing $
    md = re.sub(r"\s+\$(?=[,.\s;:\)\]])", "$", md)

    # 3. \vert → | everywhere in inline math $...$
    def fix_inline(m):
        inner = m.group(1).replace("\\vert", "|")
        return "$" + inner + "$"
    md = re.sub(r"\$([^$]+?)\$", fix_inline, md)

    # 4. \vert → | in display math $$...$$
    def fix_display(m):
        inner = m.group(1).replace("\\vert", "|")
        return "$$" + inner + "$$"
    md = re.sub(r"\$\$(.+?)\$\$", fix_display, md, flags=re.DOTALL)

    # 5. In table rows, bare | in $...$ can confuse pandoc table parser.
    #    Replace | inside $...$ with \mid ONLY inside table rows.
    lines = md.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and "$" in line:
            def protect_pipe(m):
                inner = m.group(1)
                inner = re.sub(r"(?<!\\)\|", r"\\mid ", inner)
                return "$" + inner + "$"
            lines[i] = re.sub(r"\$([^$]+?)\$", protect_pipe, line)
    md = "\n".join(lines)

    return md


def convert_one(name: str) -> bool:
    """Convert one paper. Returns True on success."""
    md_path = PAPER_DIR / f"{name}.md"
    pdf_path = PAPER_DIR / f"{name}.pdf"
    tmp_path = PAPER_DIR / f"_tmp_{name}.md"

    if not md_path.exists():
        print(f"  ❌ {md_path.name} not found")
        return False

    print(f"\n📄 {md_path.name}")
    text = md_path.read_text(encoding="utf-8")
    text = preprocess(text)
    tmp_path.write_text(text, encoding="utf-8")

    cmd = [
        "pandoc", str(tmp_path),
        "-o", str(pdf_path),
        "--pdf-engine=xelatex",
        "-V", "mainfont=Microsoft JhengHei",
        "-V", "monofont=Consolas",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
        "-V", "colorlinks=true",
        "-V", "linkcolor=blue",
        "--number-sections",
        "-V", r"header-includes=\usepackage{amsmath}\usepackage{amssymb}",
    ]

    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tmp_path.unlink(missing_ok=True)

    if r.returncode != 0:
        err_lines = [l for l in r.stderr.splitlines()
                     if l.startswith("!") or l.startswith("l.")]
        print(f"  ❌ Direct pandoc failed:")
        for el in err_lines[:6]:
            print(f"     {el}")

        # Fallback: generate .tex → repair → xelatex
        print("  🔄 Attempting .tex repair fallback...")
        return convert_via_tex_repair(name, text)
    else:
        kb = pdf_path.stat().st_size // 1024
        print(f"  ✅ {pdf_path.name} ({kb} KB)")
        return True


def convert_via_tex_repair(name: str, md_text: str) -> bool:
    """Fallback: generate .tex, patch broken math, compile with xelatex."""
    tex_path = PAPER_DIR / f"_tmp_{name}.tex"
    tmp_md = PAPER_DIR / f"_tmp2_{name}.md"
    pdf_path = PAPER_DIR / f"{name}.pdf"
    tmp_md.write_text(md_text, encoding="utf-8")

    # Generate .tex
    cmd = [
        "pandoc", str(tmp_md), "-o", str(tex_path), "--standalone",
        "-V", "mainfont=Microsoft JhengHei",
        "-V", "monofont=Consolas",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
        "-V", "colorlinks=true",
        "--number-sections",
        "-V", r"header-includes=\usepackage{amsmath}\usepackage{amssymb}",
    ]
    subprocess.run(cmd, capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
    tmp_md.unlink(missing_ok=True)

    if not tex_path.exists():
        print("  ❌ .tex generation failed")
        return False

    tex = tex_path.read_text(encoding="utf-8")

    # ---- Repair LaTeX math fragments left outside math mode ----
    # Pattern: bare \vert outside math mode → wrap in $...$
    tex = re.sub(r"(?<![\\$])\\vert\b", r"$|$", tex)

    # Pattern: \to \min outside math
    tex = re.sub(r"(?<!\$)\\to\s+\\min(?!\$)", r"$\\to \\min$", tex)

    # Pattern: \Sigma\Gamma... outside math
    tex = re.sub(r"(?<!\$)(\\Sigma\\Gamma[^\s,.)]*)", r"$\1$", tex)

    # Pattern: broken inline math \( that got split by pandoc table parsing
    # e.g. \Gamma\_\{ij\}\vert \(,  →  fix by wrapping properly
    tex = re.sub(
        r"\\Gamma\\_\\{([^}]*)\\}\\vert\s*\\\(",
        r"$\\Gamma_{\1}|$ (",
        tex,
    )

    # Fix orphan \) followed by \Sigma etc
    tex = re.sub(r"\\\)(\\Sigma)", r"$\1", tex)

    tex_path.write_text(tex, encoding="utf-8")

    # Compile with xelatex (two passes for TOC)
    for pass_n in range(2):
        r = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", str(tex_path)],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=str(PAPER_DIR),
        )

    out_pdf = PAPER_DIR / f"_tmp_{name}.pdf"
    if out_pdf.exists():
        if pdf_path.exists():
            pdf_path.unlink()
        out_pdf.rename(pdf_path)
        kb = pdf_path.stat().st_size // 1024
        print(f"  ✅ {pdf_path.name} ({kb} KB) [via .tex repair]")
        # Cleanup
        for ext in [".tex", ".aux", ".log", ".out", ".toc"]:
            p = PAPER_DIR / f"_tmp_{name}{ext}"
            p.unlink(missing_ok=True)
        return True
    else:
        for line in r.stdout.splitlines():
            if line.startswith("!"):
                print(f"     {line}")
        print("  ❌ .tex repair also failed")
        tex_path.unlink(missing_ok=True)
        return False


def main():
    print("=" * 60)
    print("ALICE Paper Series → PDF Converter (v2)")
    print("=" * 60)

    ok, fail = [], []
    for name in PAPERS:
        if convert_one(name):
            ok.append(name)
        else:
            fail.append(name)

    print("\n" + "=" * 60)
    print(f"Results: {len(ok)}/{len(ok)+len(fail)}")
    for s in ok:
        print(f"  ✅ {s}.pdf")
    for f in fail:
        print(f"  ❌ {f}")
    print("=" * 60)

    # Cleanup any remaining temp files
    for f in PAPER_DIR.glob("_tmp_*"):
        f.unlink(missing_ok=True)
    for f in PAPER_DIR.glob("_preproc_*"):
        f.unlink(missing_ok=True)
    for f in PAPER_DIR.glob("_convert_*"):
        f.unlink(missing_ok=True)

    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
