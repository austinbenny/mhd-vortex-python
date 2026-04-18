SHELL := /bin/bash

PY := uv run python
PYTEST := uv run pytest

MESH ?= data/raw/orszag_tang_128.yaml
RUN_NAME ?= orszag_tang_128
RUN_DIR := data/final/$(RUN_NAME)

DOCS_SRC_DIR := docs/source
DOCS_BUILD_DIR := docs/build
DOCS_TEX := $(DOCS_SRC_DIR)/index.tex
DOCS_PDF := $(DOCS_BUILD_DIR)/index.pdf

SLIDES_DIR := presentation
SLIDES_TEX := $(SLIDES_DIR)/slides.tex
SLIDES_PDF := $(SLIDES_DIR)/slides.pdf

LATEXMK := latexmk
LATEXMK_FLAGS := -pdf -synctex=1 -interaction=nonstopmode -halt-on-error -shell-escape

export TEXINPUTS := .:$(DOCS_SRC_DIR):
export BIBINPUTS := .:$(DOCS_SRC_DIR):
export PYTHONPATH := .

PY_SRC := $(shell find vortex scripts -name '*.py' -not -path '*/__pycache__/*')
MESH_FIG := data/final/mesh_preview/figs/mesh_preview.pdf
RUN_FIG := $(RUN_DIR)/figs/rho_final.pdf
CONV_FIG := data/final/convergence/figs/convergence.pdf

.PHONY: help all mesh solve post convergence test docs slides clean package

help:
	@echo "Targets:"
	@echo "  mesh         Render a mesh-preview figure for MESH=$(MESH)."
	@echo "  solve        Run the solver on MESH=$(MESH) into $(RUN_DIR)."
	@echo "  post         Post-process snapshots -> $(RUN_DIR)/figs."
	@echo "  convergence  Run the 32/64/128 vs 256 convergence study."
	@echo "  test         Run the unit test suite."
	@echo "  docs         Build the LaTeX report ($(DOCS_PDF))."
	@echo "  slides       Build the Beamer presentation ($(SLIDES_PDF))."
	@echo "  all          Run mesh + solve + post + convergence + docs + slides."
	@echo "  clean        Remove build artifacts."
	@echo "  package      Create package.zip of deliverables."
	@echo ""
	@echo "Override MESH=path/to/config.yaml and RUN_NAME=... to use a different config."

all: mesh solve post convergence docs slides

# Mesh preview figure — depends only on the YAML config and the scripts.
mesh: $(MESH_FIG)

$(MESH_FIG): $(MESH) $(PY_SRC)
	$(PY) -m scripts.plot_mesh $(MESH) --out $@

# Run the solver.
solve: $(RUN_DIR)/snap_final.npz

$(RUN_DIR)/snap_final.npz: $(MESH) $(PY_SRC)
	$(PY) -m scripts.run_orszag_tang $(MESH)

# Post-process snapshots into figures.
post: $(RUN_FIG)

$(RUN_FIG): $(RUN_DIR)/snap_final.npz $(PY_SRC)
	$(PY) -m scripts.plot_orszag_tang $(RUN_DIR)

# Convergence study (runs 32/64/128/256 as needed).
convergence: $(CONV_FIG)

$(CONV_FIG): $(PY_SRC)
	$(PY) -m scripts.convergence_study

test:
	$(PYTEST) tests/ -q

# LaTeX report.
docs: $(DOCS_PDF)

$(DOCS_PDF): $(DOCS_TEX) $(DOCS_SRC_DIR)/refs.bib $(RUN_FIG) $(CONV_FIG) $(MESH_FIG)
	@mkdir -p $(DOCS_BUILD_DIR)
	PATH="$$(pwd)/.venv/bin:$$PATH" $(LATEXMK) $(LATEXMK_FLAGS) -outdir=$(DOCS_BUILD_DIR) $(DOCS_TEX)

# Beamer presentation.
slides: $(SLIDES_PDF)

$(SLIDES_PDF): $(SLIDES_TEX) $(RUN_FIG) $(CONV_FIG) $(MESH_FIG)
	$(MAKE) -C $(SLIDES_DIR)

clean:
	rm -rf $(DOCS_BUILD_DIR)
	rm -rf data/interim
	$(MAKE) -C $(SLIDES_DIR) clean || true
	rm -f package.zip

package:
	@rm -f package.zip
	zip -r package.zip docs vortex scripts meshes tests presentation Makefile pyproject.toml uv.lock \
	-x "*/__pycache__/*" \
	-x "*/.DS_Store" \
	-x "docs/build/*" \
	-x "presentation/slides.aux" \
	-x "presentation/slides.log" \
	-x "presentation/slides.out" \
	-x "presentation/slides.toc" \
	-x "presentation/slides.nav" \
	-x "presentation/slides.snm" \
	-x "presentation/slides.vrb" \
	-x "presentation/slides.synctex.gz"
