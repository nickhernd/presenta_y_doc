# ==============================================================================
# Makefile -- 3DGS Report LaTeX Build
# ==============================================================================
MAIN     = main
LATEX    = pdflatex
BIBTEX   = bibtex
FLAGS    = -interaction=nonstopmode -halt-on-error

.PHONY: all clean cleanall view

# Full build: pdflatex -> bibtex -> pdflatex x2
all:
	$(LATEX) $(FLAGS) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(FLAGS) $(MAIN).tex
	$(LATEX) $(FLAGS) $(MAIN).tex

# Quick single-pass build (no bibliography update)
quick:
	$(LATEX) $(FLAGS) $(MAIN).tex

# Open the PDF
view:
	xdg-open $(MAIN).pdf &

# Remove auxiliary files (keep PDF)
clean:
	rm -f *.aux *.log *.toc *.out *.bbl *.blg *.lof *.lot *.fls *.fdb_latexmk
	rm -f sections/*.aux

# Remove everything including PDF
cleanall: clean
	rm -f $(MAIN).pdf
