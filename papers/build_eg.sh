#!/bin/bash
# Official Computer Graphics Forum submission build. The style package in
# egstyle/ is the unmodified December 2024 egPublStyle-cgf from eg.org, plus
# comment.sty, lastpage.sty and placeins.sty (public domain, from CTAN) which
# this TeX Live install does not ship.
set -e
cd "$(dirname "$0")"
export TEXINPUTS=".:./egstyle//:" BSTINPUTS=".:./egstyle//:" BIBINPUTS=".:./egstyle//:"
if [ -f ../scripts/paper_numbers.py ]; then (cd .. && python scripts/paper_numbers.py); else echo "WARN: scripts/paper_numbers.py not found; using existing numbers.tex" >&2; fi
pdflatex -interaction=nonstopmode fa_eg.tex >/dev/null
bibtex fa_eg >/dev/null
pdflatex -interaction=nonstopmode fa_eg.tex >/dev/null
pdflatex -interaction=nonstopmode fa_eg.tex >/dev/null
# a third pass settles float numbers once the appendix moves after the references
pdflatex -interaction=nonstopmode fa_eg.tex >/dev/null
echo "fa_eg.pdf  ($(grep -aci undefined fa_eg.log) undefined, $(grep -aci overfull fa_eg.log) overfull)"
