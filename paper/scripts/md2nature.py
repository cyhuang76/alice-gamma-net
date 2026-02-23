"""Convert ALICE Papers I/II/III/IV from Markdown → Nature single-column LaTeX + PDF.

Produces:
  paper/latex/Paper_I_nature.tex  +  .pdf
  paper/latex/Paper_II_nature.tex +  .pdf
  paper/latex/Paper_III_nature.tex + .pdf
  paper/latex/Paper_IV_nature.tex +  .pdf

Usage:
  python md2nature.py          # all four
  python md2nature.py 1        # Paper I only
"""
import re, sys, subprocess, shutil
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
LATEX_DIR = PAPER_DIR / "latex"

PAPERS = {
    "1": ("Paper_I_Theory",
          "The Minimum Reflection Principle — Core Theory, Mathematical "
          "Foundations, and Γ as Universal Currency"),
    "2": ("Paper_II_Architecture",
          "From Coaxial Cables to Cognition — Body-Brain Architecture, "
          "Seven-Layer Pipeline, and O(1) Perception"),
    "3": ("Paper_III_Lifecycle",
          "The Lifecycle Equation — Fontanelle Thermodynamics, Emergent "
          "Psychopathology, and Coffin-Manson Aging"),
    "4": ("Paper_IV_Emergence",
          "Emergence — Language Physics, Social Impedance Coupling, "
          "Consciousness, and the Impedance Bridge"),
}

AUTHOR = "Hsi-Yu Huang"
AUTHOR_CN = "黃璽宇"
AFFILIATION = "Independent Researcher, Taiwan"
EMAIL = "llc.y.huangll@gmail.com"

# ─── Nature-style preamble ────────────────────────────────────────
PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

%% ── Fonts (XeLaTeX) ──────────────────────────────────────────────
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Consolas}
\usepackage{xeCJK}
\setCJKmainfont{Microsoft JhengHei}

%% ── Page geometry (Nature single-column) ─────────────────────────
\usepackage[
  a4paper,
  top=2.5cm, bottom=2.5cm,
  left=2.5cm, right=2.5cm
]{geometry}

%% ── Packages ─────────────────────────────────────────────────────
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{enumitem}
\usepackage{float}
\usepackage[dvipsnames]{xcolor}
\usepackage[
  colorlinks=true,
  linkcolor=RoyalBlue,
  citecolor=OliveGreen,
  urlcolor=BrickRed
]{hyperref}
\usepackage{caption}
\usepackage{setspace}
\usepackage{titlesec}

%% ── Line spacing (Nature: ~1.5) ─────────────────────────────────
\onehalfspacing

%% ── Section heading style (Nature-like) ──────────────────────────
\titleformat{\section}
  {\large\bfseries\sffamily}{\thesection}{0.8em}{}
\titleformat{\subsection}
  {\normalsize\bfseries\sffamily}{\thesubsection}{0.6em}{}
\titleformat{\subsubsection}
  {\normalsize\itshape\sffamily}{\thesubsubsection}{0.6em}{}

%% ── Caption style ────────────────────────────────────────────────
\captionsetup{
  font={small,sf},
  labelfont={bf},
  format=plain,
  margin=1cm
}

%% ── Figure search path ──────────────────────────────────────────
\graphicspath{{../}}

%% ── Paragraph style ─────────────────────────────────────────────
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.6em}

\begin{document}
"""


# ─── Markdown → LaTeX conversion ──────────────────────────────────

def md_to_latex(md: str) -> str:
    """Convert markdown body content to LaTeX."""
    # Pre-processing: fix math escapes
    md = md.replace("\\\\\\vert", "|")
    # Also handle \vert in all math
    def _fix_vert_inline(m):
        return "$" + m.group(1).replace("\\vert", "|") + "$"
    md = re.sub(r"\$([^$]+?)\$", _fix_vert_inline, md)

    # Fix space before closing $
    md = re.sub(r"\s+\$(?=[,.\s;:\)\]])", "$", md)

    lines = md.split("\n")
    out = []
    in_code = False
    in_table = False
    in_blockquote = False
    table_rows = []
    bq_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # ── Code blocks ──
        if line.strip().startswith("```"):
            if not in_code:
                lang = line.strip()[3:].strip()
                out.append(r"\begin{verbatim}")
                in_code = True
            else:
                out.append(r"\end{verbatim}")
                in_code = False
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        # ── Horizontal rule ──
        if line.strip() == "---":
            i += 1
            continue

        # ── Blockquote ──
        if line.startswith("> ") or line.strip() == ">":
            content = line[2:] if line.startswith("> ") else ""
            bq_lines.append(content)
            i += 1
            continue
        elif bq_lines:
            bq_text = "\n".join(bq_lines)
            bq_text = convert_inline(bq_text)
            out.append(r"\begin{quote}")
            out.append(bq_text)
            out.append(r"\end{quote}")
            bq_lines = []

        # ── Tables ──
        if line.strip().startswith("|") and "|" in line[1:]:
            table_rows.append(line)
            i += 1
            continue
        elif table_rows:
            flush_table(table_rows, out)
            table_rows = []

        # ── Headings ──
        hm = re.match(r"^(#{1,4})\s+(.+)$", line)
        if hm:
            level = len(hm.group(1))
            title = convert_inline(hm.group(2))
            # Strip numbering like "1. " or "4A."
            title = re.sub(r"^\d+[A-Za-z]?\.\s*", "", title)
            cmds = {1: "section", 2: "subsection",
                    3: "subsubsection", 4: "paragraph"}
            cmd = cmds.get(level, "paragraph")
            out.append(f"\\{cmd}{{{title}}}")
            i += 1
            continue

        # ── Images ──
        img = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if img:
            caption = convert_inline(img.group(1))
            path = img.group(2)
            out.append(r"\begin{figure}[H]")
            out.append(r"\centering")
            out.append(f"\\includegraphics[width=0.85\\textwidth]{{{path}}}")
            out.append(f"\\caption{{{caption}}}")
            out.append(r"\end{figure}")
            i += 1
            continue

        # ── Bullet lists ──
        bm = re.match(r"^(\s*)-\s+(.+)$", line)
        if bm:
            items = []
            while i < len(lines):
                bm2 = re.match(r"^(\s*)-\s+(.+)$", lines[i])
                if bm2:
                    items.append(convert_inline(bm2.group(2)))
                    i += 1
                elif lines[i].strip() == "":
                    i += 1
                    break
                else:
                    break
            out.append(r"\begin{itemize}[nosep]")
            for item in items:
                out.append(f"  \\item {item}")
            out.append(r"\end{itemize}")
            continue

        # ── Numbered lists ──
        nm = re.match(r"^(\s*)\d+\.\s+(.+)$", line)
        if nm:
            items = []
            while i < len(lines):
                nm2 = re.match(r"^(\s*)\d+\.\s+(.+)$", lines[i])
                if nm2:
                    items.append(convert_inline(nm2.group(2)))
                    i += 1
                elif lines[i].strip() == "":
                    i += 1
                    break
                else:
                    break
            out.append(r"\begin{enumerate}[nosep]")
            for item in items:
                out.append(f"  \\item {item}")
            out.append(r"\end{enumerate}")
            continue

        # ── Display math (standalone $$...$$) ──
        if line.strip().startswith("$$"):
            math_lines = [line]
            if not line.strip().endswith("$$") or line.strip() == "$$":
                i += 1
                while i < len(lines):
                    math_lines.append(lines[i])
                    if lines[i].strip().endswith("$$"):
                        break
                    i += 1
            math_block = "\n".join(math_lines)
            math_block = math_block.strip().strip("$").strip()
            math_block = fix_math(math_block)
            out.append(r"\begin{equation*}")
            out.append(math_block)
            out.append(r"\end{equation*}")
            i += 1
            continue

        # ── Regular paragraph ──
        if line.strip():
            out.append(convert_inline(line))
        else:
            out.append("")

        i += 1

    # Flush remaining
    if bq_lines:
        bq_text = convert_inline("\n".join(bq_lines))
        out.append(r"\begin{quote}")
        out.append(bq_text)
        out.append(r"\end{quote}")
    if table_rows:
        flush_table(table_rows, out)

    return "\n".join(out)


def fix_math(s: str) -> str:
    """Fix math content for LaTeX."""
    s = s.replace("\\vert", "|")
    return s


def convert_inline(s: str) -> str:
    """Convert inline markdown → LaTeX."""
    # Bold + italic
    s = re.sub(r"\*\*\*(.+?)\*\*\*", r"\\textbf{\\textit{\1}}", s)
    # Bold
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    # Italic
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\textit{\1}", s)
    # Inline code
    s = re.sub(r"`([^`]+)`", r"\\texttt{\1}", s)
    # Links [text](url)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\\href{\2}{\1}", s)
    # Fix special chars in non-math text
    # Protect $ math first
    parts = re.split(r"(\$\$?[^$]+?\$\$?)", s)
    result = []
    for part in parts:
        if part.startswith("$"):
            # It's math — fix \vert, keep as-is
            part = part.replace("\\vert", "|")
            result.append(part)
        else:
            # Escape LaTeX specials
            part = part.replace("&", r"\&")
            # Don't escape # in \section etc
            part = re.sub(r"(?<!\\)#", r"\#", part)
            # Bare _ outside math → \_
            part = re.sub(r"(?<!\\)_", r"\_", part)
            # Unicode symbols
            part = part.replace("Γ", r"$\Gamma$")
            part = part.replace("Σ", r"$\Sigma$")
            part = part.replace("Φ", r"$\Phi$")
            part = part.replace("Θ", r"$\Theta$")
            part = part.replace("→", r"$\rightarrow$")
            part = part.replace("←", r"$\leftarrow$")
            part = part.replace("≥", r"$\geq$")
            part = part.replace("≤", r"$\leq$")
            part = part.replace("≈", r"$\approx$")
            part = part.replace("×", r"$\times$")
            part = part.replace("∞", r"$\infty$")
            part = part.replace("—", "---")
            part = part.replace("–", "--")
            part = part.replace("…", r"\ldots ")
            part = part.replace("§", r"\S ")
            part = part.replace("²", r"$^2$")
            part = part.replace("³", r"$^3$")
            part = part.replace("°", r"$^\circ$")
            part = part.replace("±", r"$\pm$")
            part = part.replace("ᵢ", r"$_i$")
            part = part.replace("%", r"\%")
            result.append(part)
    return "".join(result)


def flush_table(rows: list, out: list):
    """Convert collected table rows to LaTeX longtable."""
    if len(rows) < 2:
        for r in rows:
            out.append(convert_inline(r))
        return

    # Parse cells — protect math content before splitting by |
    parsed = []
    sep_idx = None
    for ri, row in enumerate(rows):
        # Temporarily replace | inside $...$ with a placeholder
        protected = _protect_math_pipes(row)
        cells = [c.strip() for c in protected.strip().strip("|").split("|")]
        # Restore placeholders
        cells = [c.replace("\x00PIPE\x00", "|") for c in cells]

        if all(re.match(r"^[-:]+$", c) for c in cells):
            sep_idx = ri
            continue
        parsed.append(cells)

    if not parsed:
        return

    ncols = max(len(r) for r in parsed)
    col_spec = " ".join(["l"] * ncols)

    out.append(r"\begin{longtable}{" + col_spec + "}")
    out.append(r"\toprule")

    for ri, cells in enumerate(parsed):
        # Pad short rows
        while len(cells) < ncols:
            cells.append("")
        row_tex = " & ".join(convert_inline(c) for c in cells)
        out.append(row_tex + r" \\")
        if ri == 0:
            out.append(r"\midrule")

    out.append(r"\bottomrule")
    out.append(r"\end{longtable}")


def _protect_math_pipes(line: str) -> str:
    """Replace | inside $...$ with placeholder so table split works."""
    result = []
    in_math = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "$":
            in_math = not in_math
            result.append(ch)
        elif ch == "|" and in_math:
            result.append("\x00PIPE\x00")
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def extract_abstract(md: str) -> str:
    """Extract abstract paragraph."""
    m = re.search(r"##\s*Abstract\s*\n+(.+?)(?=\n---|\n##)", md, re.DOTALL)
    if m:
        text = m.group(1).strip()
        text = re.sub(r"\*\*Keywords?:?\*\*.*$", "", text,
                      flags=re.DOTALL).strip()
        return convert_inline(text)
    return ""


def extract_keywords(md: str) -> str:
    """Extract keywords line."""
    m = re.search(r"\*\*Keywords?:?\*\*\s*(.+?)$", md, re.MULTILINE)
    if m:
        return convert_inline(m.group(1).strip())
    return ""


def extract_references(md: str) -> list:
    """Extract numbered references [1] ... [n]."""
    refs = []
    for m in re.finditer(r"^\[(\d+)\]\s+(.+)$", md, re.MULTILINE):
        refs.append((int(m.group(1)), convert_inline(m.group(2))))
    return sorted(refs, key=lambda x: x[0])


def extract_body(md: str) -> str:
    """Extract body between Abstract and References."""
    # Remove everything before first ## after Abstract
    parts = re.split(r"\n---\s*\n", md)

    # Find the body start (after abstract) and end (before references)
    body_start = None
    body_end = None

    lines = md.split("\n")
    found_abstract_end = False

    for i, line in enumerate(lines):
        if line.strip().startswith("## Abstract"):
            # Skip abstract — find next section
            continue
        if not found_abstract_end and re.match(r"^##\s+\d", line):
            body_start = i
            found_abstract_end = True
        if line.strip().startswith("## References"):
            body_end = i
            break

    if body_start is None:
        body_start = 0
    if body_end is None:
        body_end = len(lines)

    body_md = "\n".join(lines[body_start:body_end])

    # Also remove the ETHICAL NOTICE block (blockquote at top)
    body_md = re.sub(
        r">\s*##\s*ETHICAL NOTICE.*?(?=\n[^>]|\n\n[^>])",
        "", body_md, flags=re.DOTALL
    )

    return body_md


def build_tex(paper_id: str) -> str:
    """Build complete Nature-style .tex for one paper."""
    name, title = PAPERS[paper_id]
    md_path = PAPER_DIR / f"{name}.md"
    md = md_path.read_text(encoding="utf-8")

    abstract = extract_abstract(md)
    keywords = extract_keywords(md)
    refs = extract_references(md)
    body_md = extract_body(md)
    body_tex = md_to_latex(body_md)

    # ── Assemble .tex ──
    tex = PREAMBLE

    # ── Cover page ──
    series_line = r"\Gamma\text{-Net ALICE Research Monograph Series}"
    paper_labels = {"1": "Paper I", "2": "Paper II", "3": "Paper III"}
    paper_label = paper_labels.get(paper_id, f"Paper {paper_id}")

    tex += f"""
%% ── Cover Page ───────────────────────────────────────────────────
\\thispagestyle{{empty}}
\\begin{{titlepage}}
\\centering

\\vspace*{{3cm}}

{{\\large\\sffamily ${series_line}$}}

\\vspace{{1.5cm}}

{{\\huge\\bfseries\\sffamily {convert_inline(title)}}}

\\vspace{{2cm}}

{{\\Large {AUTHOR} ({AUTHOR_CN})}}

\\vspace{{0.8em}}

{{\\normalsize\\itshape {AFFILIATION}}}\\\\[0.4em]
{{\\normalsize Correspondence: \\href{{mailto:{EMAIL}}}{{{EMAIL}}}}}

\\vspace{{2cm}}

{{\\large {paper_label} of 3}}

\\vspace{{0.5em}}

{{\\normalsize February 2026}}

\\vfill

{{\\small\\sffamily License: Apache 2.0 \\quad|\\quad GitHub: \\href{{https://github.com/cyhuang76/alice-gamma-net}}{{cyhuang76/alice-gamma-net}}}}

\\end{{titlepage}}

\\setcounter{{page}}{{1}}

%% ── Abstract ─────────────────────────────────────────────────────
\\begin{{abstract}}
{abstract}
\\end{{abstract}}
"""
    if keywords:
        tex += f"""
\\noindent\\textbf{{Keywords:}} {keywords}

\\noindent\\rule{{\\textwidth}}{{0.4pt}}
\\vspace{{1em}}
"""
    else:
        tex += r"""
\noindent\rule{\textwidth}{0.4pt}
\vspace{1em}
"""

    # Body
    tex += "\n%% ── Body ──────────────────────────────────────────────────────────\n"
    tex += body_tex

    # References
    if refs:
        tex += r"""

\section*{References}
\begin{enumerate}[label={[\arabic*]},nosep,leftmargin=2em]
"""
        for num, text in refs:
            tex += f"  \\item {text}\n"
        tex += r"\end{enumerate}" + "\n"

    # Footer
    tex += f"""
\\vfill
\\noindent\\rule{{\\textwidth}}{{0.4pt}}

{{\\small\\itshape This is Paper {paper_id} of the \\textsf{{Γ-Net ALICE Research Monograph Series}}.}}

\\end{{document}}
"""
    return tex


def compile_tex(tex_path: Path) -> bool:
    """Compile .tex → .pdf with xelatex (2 passes for refs)."""
    for pass_n in (1, 2):
        r = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", str(tex_path.name)],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=str(tex_path.parent),
        )
    pdf = tex_path.with_suffix(".pdf")
    if pdf.exists() and pdf.stat().st_size > 1000:
        return True

    # Print errors
    for line in r.stdout.splitlines():
        if line.startswith("!"):
            print(f"     {line}")
    return False


def convert_one(paper_id: str) -> bool:
    """Convert one paper end-to-end."""
    name, title = PAPERS[paper_id]
    md_path = PAPER_DIR / f"{name}.md"

    if not md_path.exists():
        print(f"  [FAIL] {md_path.name} not found")
        return False

    print(f"\n  [{name}]")

    # Build .tex
    tex = build_tex(paper_id)
    LATEX_DIR.mkdir(exist_ok=True)
    tex_name = f"Paper_{paper_id}_nature"
    tex_path = LATEX_DIR / f"{tex_name}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"    .tex generated ({len(tex):,} chars)")

    # Copy figures to latex/ dir so \graphicspath finds them
    for fig in PAPER_DIR.glob("fig*.png"):
        dst = LATEX_DIR / fig.name
        if not dst.exists() or dst.stat().st_mtime < fig.stat().st_mtime:
            shutil.copy2(fig, dst)

    # Compile
    ok = compile_tex(tex_path)
    pdf_path = tex_path.with_suffix(".pdf")

    if ok:
        kb = pdf_path.stat().st_size // 1024
        print(f"    PDF: {pdf_path.name} ({kb} KB)")
    else:
        print(f"    [FAIL] xelatex compilation failed")

    # Cleanup aux
    for ext in (".aux", ".log", ".out", ".toc"):
        p = LATEX_DIR / f"{tex_name}{ext}"
        p.unlink(missing_ok=True)

    return ok


def main():
    print("=" * 60)
    print("  ALICE Papers -> Nature Single-Column LaTeX")
    print("=" * 60)

    targets = sys.argv[1:] if len(sys.argv) > 1 else ["1", "2", "3"]
    ok, fail = [], []

    for pid in targets:
        if pid not in PAPERS:
            print(f"  Unknown paper: {pid}")
            fail.append(pid)
            continue
        if convert_one(pid):
            ok.append(pid)
        else:
            fail.append(pid)

    print("\n" + "=" * 60)
    print(f"  Results: {len(ok)}/{len(ok)+len(fail)}")
    for s in ok:
        print(f"    OK  Paper_{s}_nature.pdf")
    for f in fail:
        print(f"    ERR Paper_{f}")
    print("=" * 60)

    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
