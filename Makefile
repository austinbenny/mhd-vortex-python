TARGET = index
SRC_DIR = docs/source
SRC = $(SRC_DIR)/$(TARGET).tex
BUILD_DIR = ./docs/build

PYTHON = .venv/bin/python
PYFLAGS = -B

LATEXMK = latexmk
LATEXMK_FLAGS = -pdf -outdir=$(BUILD_DIR) -synctex=1 -interaction=nonstopmode -halt-on-error -shell-escape

export TEXINPUTS := .:$(SRC_DIR):
export BIBINPUTS := .:$(SRC_DIR):
export PYTHONPATH := .

.PHONY: all clean package scripts docs

all: $(TARGET).pdf

data/tree.txt:
	@mkdir -p data/final
	tree -I "__pycache__|*.pyc|*.egg-info|.DS_Store" vortex scripts data/final > $@ 2>/dev/null || true

scripts:

$(TARGET).pdf: $(SRC)
	@mkdir -p $(BUILD_DIR)
	PATH="$$(pwd)/.venv/bin:$$PATH" $(LATEXMK) $(LATEXMK_FLAGS) $(SRC)

docs: $(TARGET).pdf

clean:
	rm -rf $(BUILD_DIR)
	rm -rf data/interim
	rm -f data/tree.txt
	rm -f package.zip $(TARGET).pdf $(TARGET).synctex.gz

package: clean
	@rm -f package.zip
	zip -r package.zip docs vortex scripts meshes tests Makefile pyproject.toml uv.lock \
	-x "*/__pycache__/*" \
	-x "*/.DS_Store" \
	-x "docs/build/*"
