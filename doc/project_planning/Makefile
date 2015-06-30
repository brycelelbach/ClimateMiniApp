PAPER = design
PDFTEX = pdflatex
PSTEX  = latex

default: pdf

pdf:
	-$(PSTEX) --interaction=batchmode $(PAPER)
	bibtex $(PAPER)
	-$(PSTEX) --interaction=batchmode $(PAPER)
	$(PDFTEX) $(PAPER)

dvi:
	$(PSTEX) $(PAPER)
	bibtex $(PAPER)
	$(PSTEX) $(PAPER)
	$(PSTEX) $(PAPER)

ps: dvi
	dvips -tletter -o $(PAPER).ps $(PAPER).dvi

clean:
	/bin/rm -f *.dvi *.log *.aux *.toc *.bbl *.blg *.out
	/bin/rm -f $(PAPER).pdf $(PAPER).ps
