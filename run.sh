# $pdflatex = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
# $bibtex = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\bibtex.exe"

# & $pdflatex -interaction=nonstopmode main.tex
# & $bibtex main
# & $pdflatex -interaction=nonstopmode main.tex
# & $pdflatex -interaction=nonstopmode main.tex


cd /Users/test/Documents/paper_work/PaperTex

# Install missing packages (user install, no sudo)
tlmgr init-usertree
tlmgr --usermode install silence etoolbox lineno xcolor cite caption kvoptions geometry axessibility orcidlink booktabs placeins cleveref eso-pic xstring accsupp

# Then compile
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex