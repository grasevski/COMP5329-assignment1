STUDENTIDS = 500395897_500710654

$(STUDENTIDS).zip: report/report.pdf
	mkdir -p $(STUDENTIDS)/code
	cp $< $(STUDENTIDS)/code
	cp assignment1.py $(STUDENTIDS)
	cd $(STUDENTIDS) && zip -r ../$@ *
	rm -r $(STUDENTIDS)

report/report.pdf: report/report.tex report/references.bib report/results.csv report/model.png
	cd report && pdflatex --synctex=1 report.tex && biber report && pdflatex --synctex=1 report.tex && pdflatex --synctex=1 report.tex

report/results.csv: assignment1.py
	./assignment1.py >$@

lint:
	yapf -d assignment1.py && pycodestyle assignment1.py

fix:
	yapf -i assignment1.py
