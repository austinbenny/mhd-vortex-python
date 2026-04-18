TARGET = slides
SRC = $(TARGET).tex

LATEXMK = latexmk
LATEXMK_FLAGS = -pdf -synctex=1 -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all: $(TARGET).pdf

$(TARGET).pdf: $(SRC)
	$(LATEXMK) $(LATEXMK_FLAGS) $(SRC)

clean:
	$(LATEXMK) -C $(SRC)
	rm -f *.aux *.log *.out *.toc *.synctex.gz *.nav *.snm *.vrb *.fls *.fdb_latexmk
