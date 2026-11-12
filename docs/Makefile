.PHONY: paper figure clean

paper:
	cd .. && python3 sensitivity.py > /dev/null
	cd .. && python3 policy_audit.py > /dev/null
	cd .. && python3 summarize_results.py
	cd .. && python3 verify_claims.py > results/verify_claims.txt
	cd .. && python3 make_tables.py && python3 make_figures.py
	pdflatex -interaction=nonstopmode -halt-on-error Exact-verification-paper.tex
	bibtex Exact-verification-paper
	pdflatex -interaction=nonstopmode -halt-on-error Exact-verification-paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error Exact-verification-paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error Exact-verification-paper.tex

figure:
	cd .. && python3 make_figures.py
	pdflatex -interaction=nonstopmode -halt-on-error -jobname=figure_metric_gap figure_metric_gap_standalone.tex

clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz *.blg
