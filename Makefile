STUDENTIDS = 500395897_500710654

$(STUDENTIDS).zip: report/report.pdf
	mkdir -p $(STUDENTIDS)/code
	cp $< $(STUDENTIDS)/code
	cp assignment1.py $(STUDENTIDS)
	cd $(STUDENTIDS) && zip -r ../$@ *
	rm -r $(STUDENTIDS)

report/report.pdf: report/report.tex results.csv model.png
	cd report && pdflatex report.tex

results.csv: assignment1.py
	./assignment1.py >results.csv

lint:
	yapf -d assignment1.py && pycodestyle assignment1.py
