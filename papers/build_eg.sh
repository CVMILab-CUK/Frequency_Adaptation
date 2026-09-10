#!/bin/bash
# Official Computer Graphics Forum submission build. The style package in
# egstyle/ is the unmodified December 2024 egPublStyle-cgf from eg.org, plus
# comment.sty and lastpage.sty which this TeX Live install does not ship.
set -e
cd "$(dirname "$0")"
export TEXINPUTS=".:./egstyle//:" BSTINPUTS=".:./egstyle//:" BIBINPUTS=".:./egstyle//:"
(cd .. && python scripts/paper_numbers.py)
pdflatex -interaction=nonstopmode fa_eg.tex >/dev/null
bibtex fa_eg >/dev/null
pdflatex -interaction=nonstopmode fa_eg.tex >/dev/null
pdflatex -interaction=nonstopmode fa_eg.tex >/dev/null
echo "fa_eg.pdf  ($(grep -aci undefined fa_eg.log) undefined, $(grep -aci overfull fa_eg.log) overfull)"
