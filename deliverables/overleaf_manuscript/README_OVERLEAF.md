# Overleaf Upload & Compilation Instructions

1. Upload `overleaf_research_manuscript.zip` directly in Overleaf (**New Project -> Upload Project**).
2. Set **Menu -> Compiler** to **`pdfLaTeX`** (Overleaf automatically runs **`biber`** for `biblatex` `style=apa` 7th-edition formatting).
3. `main.tex` is already authored as `Prepared by Emmanuel Kwaku Botchway` and compiles cleanly in Overleaf or locally via `python scripts/build_overleaf_manuscript.py --compile-pdf`.
4. Confirm the exported PDF renders between **4 and 10 pages** and is `< 10 MB` before uploading to Zindi alongside `research_and_modeling_code_final.zip`.
