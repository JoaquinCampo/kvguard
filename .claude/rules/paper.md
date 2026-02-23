---
paths:
  - "paper/**"
---

# Paper Conventions (ACL 2026)

- ACL format, 8 pages + unlimited references.
- Main file: `paper/main.tex`. Sections in separate files: introduction.tex, method.tex, results.tex, discussion.tex.
- Bibliography: `paper/references.bib` (BibTeX).
- Use `\citet{}` for inline citations ("Smith et al. (2025) show..."), `\citep{}` for parenthetical ("...as shown previously \citep{smith2025}").
- Tables: use `booktabs` package (`\toprule`, `\midrule`, `\bottomrule`).
- Figures: vector format (PDF) preferred. Save to `paper/figures/`.
- Numbers: use `\num{}` from siunitx for consistent formatting, or format manually with commas for thousands.
- Avoid superlatives ("novel", "groundbreaking"). Let results speak.
- Acknowledge limitations honestly in Discussion.
- Key claims from `docs/related-work-positioning.md` — read before writing positioning statements.
