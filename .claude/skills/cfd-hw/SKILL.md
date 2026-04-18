---
name: cfd-hw
description: Execute the 5-phase CFD homework workflow (Research, Implementation, Validation, LaTeX Report, Study Notes). Use when starting or continuing a CFD homework assignment.
---

# CFD Homework Workflow

Follow these 5 phases in order. If the user specifies a phase, jump to it. Otherwise start from Phase 1.

## Project Structure
- `hw/solvers.py` — core solver functions
- `hw/problem_1/part_N.py` — one script per assignment part, outputs to `data/final/part_N/`
- `Makefile` — targets for each part script, PDF depends on all scripts
- `docs/source/index.tex` — full report with theory + results + code appendix
- Run scripts via `make`, not raw python commands
- Always use the Makefile to build the PDF (`make` or `make docs`). If only the LaTeX needs re-rendering (no script changes), run just the docs target (e.g., `make docs`) to skip re-running scripts.
- Add logging in every script to create a log file that writes to the same directory as the script outputs. Delete any previously existing log file before creating a new one to avoid stale log data.

## Code Style
- Do NOT leave comments in code unless absolutely necessary for understanding
- No Unicode characters in matplotlib labels (causes LaTeX build failures)
- Keep scripts short and focused — solver logic in `solvers.py`, plotting in part scripts

## Progress Commits
- **Commit at key milestones to preserve progress.** Use git to commit after each major checkpoint: solver module complete, each part script validated, report drafted, study notes finished. This prevents losing work if something goes wrong later.
- Commit messages should be short and descriptive (e.g., "Add projection method solver", "Add part 2 steady-state results").
- Do not wait until the end to commit everything at once.

## Known Pitfalls
1. Pure central differences on collocated grids can cause checkerboard pressure decoupling — always check if smoothing or a staggered grid is needed.
2. Smoothing/dissipation coefficients that aren't scaled with grid spacing will degrade convergence on finer grids.
3. Local time stepping can cause residual oscillations on fine grids — use global dt for convergence studies.
4. Low Re cases converge very slowly with explicit time stepping — consider implicit methods or document the limitation.
5. When detecting flow features numerically (e.g., entrance length), compare against the numerical downstream value, not the analytical one.

---

## Phase 1: Research

- **Read the problem statement thoroughly before any other work.** Read the homework PDF (e.g., `homework3.pdf`) end-to-end and note every requirement, constraint, and specific instruction (e.g., "achieve second order spatial accuracy"). These details dictate scheme choices and must be satisfied — missing them leads to point deductions.
- **Compress large PDFs before reading.** If a PDF is greater than 20 MB, compress it first using `gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook -dNOPAUSE -dBATCH -sOutputFile=compressed.pdf input.pdf`, then read the compressed version. This avoids failures from the Read tool's page-size limits on large files.
- **Read ALL reference materials first and summarize every section/topic.** Before proposing an approach or writing a plan, read every PDF in `references/` section by section. List every topic covered — especially sections on stability, smoothing, dissipation, convergence criteria, or oscillation suppression. These sections often contain the exact fix for the hardest implementation problems. Do not skim; process every section even at the bottom of pages.

## Phase 2: Implementation

- Use **normalized residual** (res/res0) as primary criterion for convergence.
- **Be aware that the initial residual (res0) scales with grid size.** Finer grids produce larger initial residuals, so a fixed normalized tolerance maps to different absolute residual levels across grids. For convergence studies, use global time stepping and plateau detection as a fallback (if residual changes < 0.01% over 5000 iterations, stop).
- Default tolerance: 1e-3 normalized.
- **If convergence issues take >10 minutes to debug, document the error and move on.** Note the issue in the LaTeX report and explain what was observed. Partial results with honest documentation are better than no results.

## Phase 3: Validation

- **Run scripts one at a time and check outputs before proceeding to the next.** Do not batch-run all scripts. After each script finishes, read its stdout/stderr and inspect generated plots/data before running the next one.
- **Always inspect every plot and table output.** After running a script, read the output and verify:
  - Correct axis labels and ranges
  - Physical plausibility (e.g., velocity profiles should be parabolic for Poiseuille flow)
  - Convergence behavior (residual should decrease monotonically or plateau)
  - Quantitative agreement with analytical solutions where available

## Phase 4: LaTeX Report

- Match the previous hw style: straightforward explanations, `align` environments for equations, `[H]` figure placement, `\Cref` for references
- Write in a direct and simple-language style — avoid flowery introductions or AI-sounding language.
- When talking about results be explanatory but not verbose. Use bullet points where appropriate.
- **Interpret every plot and figure in the report text.** Do not just say "Figure X shows the results." Describe what the plot shows physically, point out key features (e.g., vortex location, boundary layer thinning, convergence behavior), and explain whether the results match expectations. The reader should understand the significance without needing to study the figure themselves.
- Do not use bold text in the report body. No `\textbf{}` for emphasis in prose — let the structure (sections, itemize) do the work.
- Do not add citations unless the user explicitly asks for them
- Include code listings in appendix via `\inputminted`
- Every rubric item must be addressed in a specific section
- **Show all intermediate steps in math derivations.** Do not skip steps — write out every algebraic manipulation, substitution, and simplification so the reader can follow the derivation line by line. If a step involves a non-trivial identity or cancellation, state it explicitly.
- Math notation consistency: Use the same LaTeX symbols and notation that appear in the study notes (`references/notes.md`). The report and notes should use identical variable names, subscripts, and equation forms so they read as a unified set of documents.

## Phase 5: Study Notes

- After the report is finalized, create study notes in `references/notes.md` that break down every key concept from the homework in simple terms with examples and pseudocode.
- **No Unicode characters in notes** — use only ASCII-safe alternatives (e.g., "Correct." instead of a checkmark). Unicode symbols break `pdflatex` and render as blanks with `xelatex` + standard fonts.
- **PDF-first formatting:** Write the notes to render well as a PDF. Avoid very wide tables (keep to 3-4 columns max or use short cell text), use `---` for horizontal rules in tables (not long dashes), and prefer stacked lists over wide comparison tables when content is wordy.
- Convert to PDF via `pandoc references/notes.md --standalone --pdf-engine=pdflatex -V geometry:margin=1in -V fontsize=11pt -o references/notes.pdf`.
- Notes should cover: the physical problem, method motivation, governing equations, grid setup, discretization schemes, boundary conditions, time stepping, numerical fixes (e.g., pressure smoothing), convergence criteria, and analytical solutions.
- Mention both what the lecture notes prescribe and what was actually implemented, noting any differences and why.
- **Math notation consistency:** Use the same symbols and notation as the LaTeX report (`docs/source/index.tex`). The notes and report should be interchangeable in their math — same variable names, subscripts, and equation forms.
