#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Stitch latest main/supp sources into submission bodies.
sed -n '69,479p' main.tex > main_body.tex
sed -n '33,212p' supp.tex \
  | sed 's/\\label{sec:ablation}/\\label{supp:sec:ablation}/g; s/\\ref{sec:ablation}/\\ref{supp:sec:ablation}/g' \
  > supp_body.tex

rm -f submit.aux submit.out submit.toc submit.bbl submit.blg

pdflatex -interaction=nonstopmode submit.tex
bibtex submit
pdflatex -interaction=nonstopmode submit.tex
pdflatex -interaction=nonstopmode submit.tex

echo "Camera-ready PDF written to submit.pdf"
