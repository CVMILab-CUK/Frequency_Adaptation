# papers/Makefile -- numbers.tex is regenerated from results/ every build, so a
# figure in the PDF can never drift from the run that produced it.
.PHONY: all clean en ko numbers
all: numbers en ko

numbers:
	cd .. && python scripts/paper_numbers.py

en: numbers
	pdflatex -interaction=nonstopmode fa_en.tex >/dev/null
	bibtex fa_en >/dev/null
	pdflatex -interaction=nonstopmode fa_en.tex >/dev/null
	pdflatex -interaction=nonstopmode fa_en.tex >/dev/null
	@echo "fa_en.pdf"

ko: numbers
	lualatex -interaction=nonstopmode fa_ko.tex >/dev/null
	bibtex fa_ko >/dev/null
	lualatex -interaction=nonstopmode fa_ko.tex >/dev/null
	lualatex -interaction=nonstopmode fa_ko.tex >/dev/null
	@echo "fa_ko.pdf"

refs:
	cd .. && python scripts/arxiv_verify.py verify $$(python -c "import json;print(' '.join(json.load(open('papers/refs_index.json'))[k]['id'] for k in json.load(open('papers/refs_index.json'))))") >/dev/null && python scripts/make_bib.py

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg
