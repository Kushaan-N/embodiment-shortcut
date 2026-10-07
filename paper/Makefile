# Build on a compute node (texlive module), e.g.:
#   srun -p cpu --time=00:10:00 --mem=4G bash -lc 'module load texlive && make -C paper'
FIGS := figures/friction.tex figures/resolution.tex figures/h2.tex
main.pdf: main.tex refs.bib $(FIGS)
	pdflatex -interaction=nonstopmode main.tex >/dev/null
	bibtex main >/dev/null
	pdflatex -interaction=nonstopmode main.tex >/dev/null
	pdflatex -interaction=nonstopmode main.tex | grep -E "Warning|Error|Output written" || true
clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
