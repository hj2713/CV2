# CV2 Final Report

This folder contains the NeurIPS-style LaTeX report for the project.

Main file:

```text
Report/main.tex
```

The report uses the final 30-image exported results and comparison panels:

```text
Code/data/outputs/results_003.csv
Code/data/outputs/run_003/
```

To compile on a machine with LaTeX installed:

```bash
cd Report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The current local machine does not have `pdflatex` installed, so the source has
been syntax-structured but not locally rendered.
