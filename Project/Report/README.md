# CV2 Final Report

This folder contains the NeurIPS-style LaTeX report for the project.

Main file:

```text
Report/main.tex
```

The report uses the latest exported results and comparison panels:

```text
Code/data/outputs/results_*.csv
Code/data/outputs/run_*/
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
