$pdflatex = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
$bibtex = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\bibtex.exe"

& $pdflatex -interaction=nonstopmode main.tex
& $bibtex main
& $pdflatex -interaction=nonstopmode main.tex
& $pdflatex -interaction=nonstopmode main.tex
