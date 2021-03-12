STUDENTIDS = 500395897_500710654

$(STUDENTIDS).zip: assignment1.py report/report.pdf
	mkdir -p $(STUDENTIDS)/code
	cp $< $(STUDENTIDS)/code
	cp $(word 2,$^) $(STUDENTIDS)
	cd $(STUDENTIDS) && zip -r ../$@ *
	rm -r $(STUDENTIDS)

report/report.pdf: report/report.tex
	cd report && pdflatex report.tex

lint:
	yapf -d assignment1.py && pycodestyle assignment1.py
