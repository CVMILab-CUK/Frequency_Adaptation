#!/bin/bash
# Eurographics 2027 Full Papers submission build. The style package in
# egstyle_eg2027/ is the unmodified EG_2027 package, plus comment.sty,
# lastpage.sty and placeins.sty (public domain, from CTAN) which neither the
# package nor this TeX Live install ships.
set -e
cd "$(dirname "$0")"
export TEXINPUTS=".:./egstyle_eg2027//:" BSTINPUTS=".:./egstyle_eg2027//:" BIBINPUTS=".:./egstyle_eg2027//:"
if [ -f ../scripts/paper_numbers.py ]; then (cd .. && python scripts/paper_numbers.py); else echo "WARN: scripts/paper_numbers.py not found; using existing numbers.tex" >&2; fi
pdflatex -interaction=nonstopmode fa_eg.tex >/dev/null
bibtex fa_eg >/dev/null
# the supplementary carries the appendix; the two documents cite each other's
# table numbers through xr, so each is compiled twice around the other
pdflatex -interaction=nonstopmode fa_supp.tex >/dev/null
pdflatex -interaction=nonstopmode fa_eg.tex >/dev/null
pdflatex -interaction=nonstopmode fa_eg.tex >/dev/null
pdflatex -interaction=nonstopmode fa_supp.tex >/dev/null
pdflatex -interaction=nonstopmode fa_eg.tex >/dev/null
echo "fa_eg.pdf  ($(grep -aci undefined fa_eg.log) undefined, $(grep -aci overfull fa_eg.log) overfull)"
echo "fa_supp.pdf  ($(grep -aci undefined fa_supp.log) undefined)"
