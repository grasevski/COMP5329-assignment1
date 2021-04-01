STUDENTIDS = 500395897_500710654

$(STUDENTIDS).zip: report/report.pdf
	mkdir -p $(STUDENTIDS)/code
	cp $< $(STUDENTIDS)/code
	cp $(word 2,$^) $(STUDENTIDS)
	cd $(STUDENTIDS) && zip -r ../$@ *
	rm -r $(STUDENTIDS)

report/report.pdf: report/report.tex report/results.csv
	cd report && pdflatex report.tex

report/results.csv: assignment1.py
	./assignment1.py >report/results.csv

lint:
	yapf -d assignment1.py && pycodestyle assignment1.py
